import asyncio
import json
import logging
import shutil
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from .class_base import (
    GPUExtractor, pca_torch, kmeans_torch, save_cluster_montages, set_seed,
    NBINS, HARMONICS, RAD_BANDS, CENTER_MASK_RADIUS, RMAX, NTHETA, BATCH,
    center_crop_cpu, preprocess_cpu_for_montage,
)
from ..workflows import data_root
from ..lock_manager import LockManager

logger = logging.getLogger(__name__)


def draw_cluster_classification_map(xs, ys, labels, num_clusters, out_path, cmap_name="tab20"):
    grid = np.full((int(np.max(xs)), int(np.max(ys))), -1, dtype=np.int32)
    grid[np.asarray(xs) - 1, np.asarray(ys) - 1] = labels
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    image = ax.imshow(grid, cmap=plt.get_cmap(cmap_name if num_clusters <= 20 else "turbo", num_clusters), vmin=-0.5, vmax=num_clusters - 0.5)
    fig.colorbar(image, ax=ax, ticks=range(0, num_clusters, max(1, num_clusters // 16)), label="Cluster")
    ax.set_xlabel("Scan column")
    ax.set_ylabel("Scan row")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_individual_cluster_images(data_array, labels_np, min_d2_np, out_dir, images_per_cluster=16):
    for cluster in np.unique(labels_np):
        indices = np.flatnonzero(labels_np == cluster)
        selected = indices[np.argsort(min_d2_np[indices])[:images_per_cluster]]
        folder = Path(out_dir) / f"cluster_{cluster}"
        folder.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(selected):
            image = preprocess_cpu_for_montage(center_crop_cpu(data_array[index]))
            plt.imsave(folder / f"individual_{rank:03d}.png", image, cmap="gray", vmin=0, vmax=1)


def _do_clustering_sync(entries, out_dir, k_clusters, seed, device_str, progress):
    import torch
    set_seed(seed)
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = GPUExtractor(device, NBINS, HARMONICS, RAD_BANDS, CENTER_MASK_RADIUS, RMAX, NTHETA, BATCH)
    parts, xs, ys, locations = [], [], [], []
    for index, entry in enumerate(entries):
        if progress:
            progress("features", 0.45 * index / len(entries), f"Extracting row {index + 1}/{len(entries)}")
        row = scipy.io.loadmat(entry["file_path"])["data"]
        def batch_progress(fraction):
            if progress:
                progress("features", 0.45 * (index + fraction) / len(entries), f"Extracting row {index + 1}/{len(entries)}")
        parts.append(extractor.extract(row, progress=batch_progress))
        xs.extend([entry["row_index"]] * len(row))
        ys.extend(range(1, len(row) + 1))
        locations.extend((entry["file_path"], col) for col in range(len(row)))
        del row
    features = np.concatenate(parts)
    del parts
    if len(features) < 2 or not 1 <= k_clusters <= len(features):
        raise ValueError("Clustering needs at least two patterns and K must not exceed the pattern count")
    if progress:
        progress("pca", 0.48, "Reducing feature dimensions")
    tensor = torch.from_numpy(features).to(device)
    components = min(50, tensor.shape[1], tensor.shape[0] - 1)
    feature_dim = tensor.shape[1]
    embedding, _, _ = pca_torch(tensor, components)
    if progress:
        progress("cluster", 0.55, "Running K-Means")
    labels, distances, _ = kmeans_torch(embedding, k_clusters, iters=50, restarts=2, seed=seed)
    labels = labels.cpu().numpy().astype(np.int32)
    distances = distances.cpu().numpy().astype(np.float32)
    del tensor, embedding, features
    out_dir = Path(out_dir)
    montage_dir = out_dir / "montages"
    individual_dir = out_dir / "individual_images"
    for index, cluster in enumerate(np.unique(labels)):
        if progress:
            progress("images", 0.65 + 0.15 * index / k_clusters, f"Saving cluster {cluster}")
        members = np.flatnonzero(labels == cluster)
        selected = members[np.argsort(distances[members])[:16]]
        images = []
        cached_path, cached_row = None, None
        for member in selected:
            path, col = locations[member]
            if path != cached_path:
                cached_row = scipy.io.loadmat(path)["data"]
                cached_path = path
            images.append(center_crop_cpu(cached_row[col]).copy())
        data = np.stack(images)
        subset_labels = np.full(len(data), cluster)
        save_cluster_montages(data, subset_labels, distances[selected], montage_dir, (4, 4))
        save_individual_cluster_images(data, subset_labels, distances[selected], individual_dir)
        del data, images, cached_row
    xs, ys = np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)
    map_path = out_dir / "cluster_map.png"
    draw_cluster_classification_map(xs, ys, labels, k_clusters, map_path)
    npz_path = out_dir / "clustering.npz"
    np.savez_compressed(npz_path, labels=labels, xs=xs, ys=ys, distances=distances)
    return {"out_dir": str(out_dir), "montage_dir": str(montage_dir),
            "individual_images_dir": str(individual_dir), "xy_map": str(map_path),
            "npz": str(npz_path), "k": k_clusters, "total_patterns": len(labels),
            "execution": {"device": str(device), "pca_components": components, "feature_dim": feature_dim,
                          "torch_version": torch.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__}}


async def store_clustering_results(sql_driver, scan_id, result, parameters, progress=None):
    with np.load(result["npz"]) as data:
        labels, xs, ys = data["labels"], data["xs"], data["ys"]
    if not len(labels) == len(xs) == len(ys):
        raise ValueError("Cluster labels and coordinates have different lengths")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= result["k"]):
        raise ValueError("Cluster labels must be integer indices from 0 to K-1")
    if not np.issubdtype(xs.dtype, np.integer) or not np.issubdtype(ys.dtype, np.integer):
        raise ValueError("Scan coordinates must be integers")
    patterns = await sql_driver.execute_query(
        "SELECT dp.id, rmf.row_index, dp.col_index FROM diffraction_patterns dp "
        "JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id WHERE rmf.scan_id = %s", [scan_id])
    identifiers = {(r.cells["row_index"], r.cells["col_index"]): r.cells["id"] for r in patterns or []}
    if len(labels) != len(identifiers) or len(set(zip(xs.tolist(), ys.tolist()))) != len(labels):
        raise ValueError("Cluster coordinates must cover the complete scan exactly once")
    async with sql_driver.transaction() as tx:
        await tx.execute_query("SELECT pg_advisory_xact_lock(%s)", [scan_id])
        run = await tx.execute_query(
            "INSERT INTO clustering_runs (scan_id, k_value, algorithm_details, results_path, parameters) "
            "VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING id",
            [scan_id, result["k"], "Polar features / PCA / K-Means", result["out_dir"], json.dumps(parameters)])
        run_id = run[0].cells["id"]
        indices, counts = np.unique(labels, return_counts=True)
        await tx.copy_rows("identified_clusters", ["run_id", "cluster_index", "pattern_count"],
                           ((run_id, int(index), int(count)) for index, count in zip(indices, counts)))
        await tx.execute_query("CREATE TEMP TABLE cluster_updates (pattern_id INTEGER PRIMARY KEY, label INTEGER) ON COMMIT DROP")
        def records():
            for index, (x, y, label) in enumerate(zip(xs, ys, labels)):
                if progress and index % 1000 == 0:
                    progress("catalog", 0.82 + 0.14 * index / len(labels), f"Saving {index}/{len(labels)} assignments")
                yield identifiers[(int(x), int(y))], int(label)
        await tx.copy_rows("cluster_updates", ["pattern_id", "label"], records())
        await tx.execute_query(
            "UPDATE diffraction_patterns dp SET cluster_label = u.label, clustering_run_id = %s "
            "FROM cluster_updates u WHERE dp.id = u.pattern_id", [run_id])
        await tx.execute_query(
            "INSERT INTO pattern_cluster_assignments (pattern_id, cluster_id) "
            "SELECT u.pattern_id, ic.id FROM cluster_updates u JOIN identified_clusters ic "
            "ON ic.run_id = %s AND ic.cluster_index = u.label", [run_id])
        await tx.execute_query("UPDATE scans SET classification_map_path = %s WHERE id = %s", [result["xy_map"], scan_id])
    result.update(clustering_run_id=run_id, updated_patterns=len(labels), classification_map_saved=bool(result["xy_map"]))
    return result


async def analyze_scan(sql_driver, scan_identifier, out_root=None, k_clusters=16, seed=0, device=None, progress=None):
    column = "id" if isinstance(scan_identifier, int) else "scan_name"
    scan = await sql_driver.execute_query(f"SELECT id, scan_name FROM scans WHERE {column} = %s", [scan_identifier])
    if not scan:
        raise ValueError(f"Scan not found: {scan_identifier}")
    scan_id = scan[0].cells["id"]
    rows = await sql_driver.execute_query(
        "SELECT id AS mat_file_id, row_index, file_path FROM raw_mat_files WHERE scan_id = %s ORDER BY row_index", [scan_id])
    entries = [r.cells for r in rows or []]
    if not entries:
        raise ValueError("Scan contains no imported rows")
    with LockManager.hold({"scan_id": scan_id, "scan_identifier": scan_identifier}):
        folder = Path(out_root or data_root() / "results") / uuid4().hex
        folder.mkdir(parents=True)
        saved = False
        parameters = {"seed": seed, "device": device or "auto", "k_clusters": k_clusters,
                      "nbins": NBINS, "harmonics": HARMONICS, "rad_bands": RAD_BANDS,
                      "center_mask_radius": CENTER_MASK_RADIUS, "rmax": RMAX, "ntheta": NTHETA,
                      "batch_size": BATCH, "pca_components": 50, "kmeans_iterations": 50,
                      "kmeans_restarts": 2, "engine_version": "0.2.0"}
        try:
            result = await asyncio.to_thread(_do_clustering_sync, entries, folder, k_clusters, seed, device, progress)
            parameters.update(result.pop("execution"))
            result["scan_id"] = scan_id
            result["scan_name"] = scan[0].cells["scan_name"]
            (folder / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
            result = await store_clustering_results(sql_driver, scan_id, result, parameters, progress)
            saved = True
            return result
        finally:
            if not saved:
                shutil.rmtree(folder, ignore_errors=True)


async def regenerate_classification_map(
    sql_driver,
    scan_identifier: int | str,
    out_path: str | Path,
    clustering_run_id: int = None,
) -> str:
    """
    Regenerate classification map using LLM-assigned classification codes.
    Patterns sharing the same classification_code use the same color family; clusters within that code get distinct shades.

    Returns the file path of the saved image.
    """
    logger = logging.getLogger(f"regenerate_map_{scan_identifier}")
    logger.info(f"Regenerating classification map for scan {scan_identifier}")
    # resolve scan_id
    if isinstance(scan_identifier, int):
        scan_id = scan_identifier
    else:
        rows = await sql_driver.execute_query(
            "SELECT id FROM scans WHERE scan_name = %s", [scan_identifier]
        )
        if not rows:
            raise RuntimeError(f"Scan not found: {scan_identifier}")
        scan_id = rows[0].cells["id"]
    # determine clustering_run_id
    if clustering_run_id is None:
        rr = await sql_driver.execute_query(
            "SELECT id FROM clustering_runs WHERE scan_id = %s ORDER BY run_timestamp DESC LIMIT 1",
            [scan_id],
        )
        if not rr:
            raise RuntimeError(f"No clustering run for scan: {scan_id}")
        clustering_run_id = rr[0].cells["id"]
    # Fetch all diffraction patterns for the clustering run (this ensures full grid coverage)
    patterns_query = (
        "SELECT rmf.row_index, dp.col_index, ic.cluster_index AS cluster_label "
        "FROM pattern_cluster_assignments pca "
        "JOIN identified_clusters ic ON pca.cluster_id = ic.id "
        "JOIN diffraction_patterns dp ON pca.pattern_id = dp.id "
        "JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id "
        "WHERE ic.run_id = %s ORDER BY rmf.row_index, dp.col_index"
    )
    pattern_rows = await sql_driver.execute_query(patterns_query, [clustering_run_id])
    if not pattern_rows:
        raise RuntimeError(
            f"No diffraction pattern records found for clustering run {clustering_run_id}"
        )

    # Load any available LLM results (may be a subset)
    llm_query = (
        "SELECT row_index, col_index, cluster_index, llm_detailed_features "
        "FROM llm_analysis_results "
        "WHERE scan_id = %s AND clustering_run_id = %s"
    )
    llm_rows = await sql_driver.execute_query(llm_query, [scan_id, clustering_run_id])

    # Build quick lookup maps
    llm_map = {}  # (row,col) -> classification_code
    cluster_code_map = {}  # cluster_index -> classification_code (if any LLM exists for cluster)
    if llm_rows:
        for r in llm_rows:
            xi = int(r.cells["row_index"])
            yi = int(r.cells["col_index"])
            info = r.cells.get("llm_detailed_features")
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except json.JSONDecodeError:
                    info = {}
            code = int(info.get("classification_code", -1))
            llm_map[(xi, yi)] = code
            try:
                cl = int(r.cells.get("cluster_index", -1))
                if cl >= 0 and code >= 0:
                    # If multiple entries for a cluster, keep the first non-negative code (or override)
                    cluster_code_map[cl] = code
            except Exception:
                pass

    xs, ys, clusters, codes = [], [], [], []
    for r in pattern_rows:
        xi = int(r.cells["row_index"])
        yi = int(r.cells["col_index"])
        cl = int(r.cells.get("cluster_label", -1))

        # Determine classification code: prefer direct LLM per-pattern, then cluster-level mapping, else -1
        code = llm_map.get((xi, yi), None)
        if code is None:
            code = cluster_code_map.get(cl, -1)

        xs.append(xi)
        ys.append(yi)
        clusters.append(cl)
        codes.append(int(code))
    xs_np = np.array(xs, dtype=int)
    ys_np = np.array(ys, dtype=int)
    clusters_np = np.array(clusters, dtype=int)
    codes_np = np.array(codes, dtype=int)
    # define color families for each classification code
    family_cmaps = {
        0: "Greys",  # Vacuum
        1: "Blues",  # Crystalline
        2: "Greens",  # Amorphous
        3: "Reds",  # Mixed-State
    }
    # Build a mapping from cluster -> representative classification code
    cluster_to_code = {}
    for cl in sorted(set(clusters_np.tolist())):
        # find codes for this cluster
        mask = clusters_np == cl
        codes_for_cluster = codes_np[mask]
        # pick the most common non-negative code if possible
        if len(codes_for_cluster) == 0:
            cluster_to_code[cl] = -1
            continue
        vals, counts = np.unique(codes_for_cluster, return_counts=True)
        # prefer non -1 codes; if none, keep -1
        nonneg_mask = vals >= 0
        if nonneg_mask.any():
            # choose the non-negative code with highest count
            valid_vals = vals[nonneg_mask]
            valid_counts = counts[nonneg_mask]
            chosen = valid_vals[np.argmax(valid_counts)]
            cluster_to_code[cl] = int(chosen)
        else:
            cluster_to_code[cl] = -1

    # Now assign an RGBA color to every cluster using interpolated gradients
    # Define light->dark color ranges for each classification code
    from matplotlib.colors import to_rgba

    def interpolate_color(light_hex, dark_hex, num_colors):
        """Interpolate between light and dark colors to create a gradient."""
        light_rgb = tuple(int(light_hex.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
        dark_rgb = tuple(int(dark_hex.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])

        colors = []
        for i in range(num_colors):
            if num_colors == 1:
                ratio = 0.5
            else:
                ratio = i / (num_colors - 1)

            r = int(light_rgb[0] + (dark_rgb[0] - light_rgb[0]) * ratio)
            g = int(light_rgb[1] + (dark_rgb[1] - light_rgb[1]) * ratio)
            b = int(light_rgb[2] + (dark_rgb[2] - light_rgb[2]) * ratio)

            colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors

    # Define color ranges for each classification code
    color_ranges = {
        0: ("#d8cdda", "#cbb8e1"),  # Vacuum: light purple to medium purple
        2: ("#b6d9f5", "#3A4686"),  # Crystalline: light green to dark green
        1: ("#b9dc69", "#31572c"),  # Amorphous: light blue to dark blue
        3: ("#e49f5e", "#AB4B31"),  # Mixed-State: light yellow to dark amber
    }

    # Build a mapping cluster -> category name expected by get_color_map
    code_labels = {
        0: "Vacuum",
        1: "Crystalline",
        2: "Amorphous",
        3: "Mixed-State",
        -1: "Unassigned",
    }

    # Group clusters by their classification code
    code_to_clusters = {}
    for cl, code in cluster_to_code.items():
        code_to_clusters.setdefault(code, []).append(cl)

    cluster_colors = {}
    for code, clusters_list in code_to_clusters.items():
        if code in color_ranges:
            # Generate interpolated colors for this code
            light_color, dark_color = color_ranges[code]
            num_clusters = len(clusters_list)
            gradient_colors = interpolate_color(light_color, dark_color, num_clusters)

            # Assign colors to clusters (sorted by cluster index)
            for i, cl in enumerate(sorted(clusters_list)):
                cluster_colors[cl] = to_rgba(gradient_colors[i])
        else:
            # Fallback for unknown codes - use gray
            for cl in clusters_list:
                cluster_colors[cl] = (0.8, 0.8, 0.8, 1.0)

    # Handle any clusters not in code_to_clusters (defensive)
    for cl in sorted(set(clusters_np.tolist())):
        if cl not in cluster_colors:
            cluster_colors[cl] = (0.8, 0.8, 0.8, 1.0)

    # Plot all clusters (use assigned colors)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    for cl in sorted(set(clusters_np.tolist())):
        mask = clusters_np == cl
        if not mask.any():
            continue
        color = cluster_colors.get(cl, (0.8, 0.8, 0.8, 1.0))  # fallback light gray
        ax.scatter(ys_np[mask], xs_np[mask], c=[color], s=1)

    # Build a detailed legend listing every cluster (K index) with its color and type
    from matplotlib.patches import Patch

    # Human readable labels for classification codes
    code_labels = {
        0: "Vacuum",
        1: "Crystalline",
        2: "Amorphous",
        3: "Mixed-State",
        -1: "Unassigned",
    }

    # We want the legend grouped by type (color family) and within each group
    # the clusters should be ordered by their K index (ascending).
    legend_handles = []
    legend_labels = []

    # Ensure we have mapping from code -> clusters (code_to_clusters already built above)
    # Determine the canonical order of codes: follow family_cmaps ordering, then any
    # unexpected positive codes, and finally unassigned (-1) if present.
    ordered_codes = [c for c in family_cmaps.keys() if c in code_to_clusters]
    # add other codes (excluding -1) in numeric order
    other_codes = sorted(
        [c for c in code_to_clusters.keys() if c not in ordered_codes and c != -1]
    )
    ordered_codes.extend(other_codes)
    if -1 in code_to_clusters:
        ordered_codes.append(-1)

    # For each code group, add legend entries for each cluster in ascending K order
    for code in ordered_codes:
        clusters_list = sorted(code_to_clusters.get(code, []))
        for cl in clusters_list:
            color = cluster_colors.get(cl, (0.8, 0.8, 0.8, 1.0))
            legend_handles.append(Patch(facecolor=color, edgecolor="none"))
            legend_labels.append(f"K{cl}: {code_labels.get(code, f'Code {code}')}")

    # If there are any clusters not present in code_to_clusters (defensive), add them at the end
    all_clusters = sorted(set(clusters_np.tolist()))
    covered = (
        set(sum([v for v in code_to_clusters.values()], []))
        if code_to_clusters
        else set()
    )
    remaining = [c for c in all_clusters if c not in covered]
    for cl in remaining:
        color = cluster_colors.get(cl, (0.8, 0.8, 0.8, 1.0))
        legend_handles.append(Patch(facecolor=color, edgecolor="none"))
        legend_labels.append(f"K{cl}: Unknown")

    if legend_handles:
        # Place legend close to the image (inside the axes, upper-right) and compact
        legend = ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.85,
            fontsize="small",
            title="Classes",
            borderaxespad=0.1,
        )
        # Ensure legend markers are clean (no edge lines)
        try:
            for handle in legend.legendHandles:
                if hasattr(handle, "set_edgecolor"):
                    handle.set_edgecolor("none")
        except Exception:
            pass

    # Force-disable any grid/axis lines and ticks to ensure a clean image
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return str(out_p)
