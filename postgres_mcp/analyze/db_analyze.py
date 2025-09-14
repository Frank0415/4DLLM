import os
import asyncio
from pathlib import Path
from typing import Union, Dict, Any, List
from tqdm import tqdm
import logging
import json
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# reuse clustering and plotting utilities from pre_class_v2
from .class_base import (
    GPUExtractor,
    pca_torch,
    kmeans_torch,
    save_cluster_montages,
    xy_plot_final,
    set_seed,
    NBINS,
    HARMONICS,
    RAD_BANDS,
    CENTER_MASK_RADIUS,
    RMAX,
    NTHETA,
    BATCH,
    CATEGORIES,
    center_crop_cpu,
    preprocess_cpu_for_montage,
)
from .type_colors import get_color_map
from ..sql import SafeSqlDriver


def get_cluster_categories(num_clusters: int) -> List[int]:
    """
    Return a list of cluster indices for a given number of clusters.
    """
    return list(range(num_clusters))


def draw_cluster_classification_map(
    xs: np.ndarray,
    ys: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    out_path: Path,
    cmap_name: str = "tab20",
) -> None:
    """
    Draw and save a classification map for all clusters with distinct colors.

    Args:
        xs: Array of X-coordinates (row indices).
        ys: Array of Y-coordinates (column indices).
        labels: Array of cluster labels for each point.
        num_clusters: Total number of clusters.
        out_path: Path to save the classification map image.
        cmap_name: Name of matplotlib colormap supporting at least num_clusters colors.
    """
    cmap = plt.get_cmap(cmap_name, num_clusters)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    for cluster in range(num_clusters):
        mask = labels == cluster
        if mask.any():
            ax.scatter(
                ys[mask], xs[mask], c=[cmap(cluster)], s=1, label=f"Cluster {cluster}"
            )
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(
        markerscale=5, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small"
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# Override MONTAGE_GRID to create 4x4 grids (16 pictures) instead of default
CLUSTER_MONTAGE_GRID = (4, 4)


def save_individual_cluster_images(
    data_array, labels_np, min_d2_np, out_dir, images_per_cluster=16
):
    """
    Save individual images for each cluster.

    Args:
        data_array: Array of diffraction patterns
        labels_np: Cluster labels for each pattern
        min_d2_np: Distance to cluster center for each pattern
        out_dir: Output directory for individual images
        images_per_cluster: Number of individual images to save per cluster
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq = sorted(np.unique(labels_np))

    for k in tqdm(uniq, desc="Saving individual cluster images"):
        idxs = np.where(labels_np == k)[0]
        if len(idxs) == 0:
            continue

        # Sort by distance to cluster center (best representatives first)
        order = np.argsort(min_d2_np[idxs])
        # Take the best images (up to images_per_cluster)
        take = (
            idxs[order[:images_per_cluster]]
            if len(order) >= images_per_cluster
            else idxs[order]
        )

        # Create a directory for this cluster's individual images
        cluster_dir = out_dir / f"cluster_{k}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Save each individual image
        for i, idx in enumerate(take):
            img = center_crop_cpu(data_array[idx], 224)
            img = preprocess_cpu_for_montage(img)

            # Save as individual image
            plt.figure(figsize=(2.24, 2.24), dpi=100)  # 224x224 pixels
            plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(
                cluster_dir / f"individual_{i:03d}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


async def _load_mat_async(path: str) -> np.ndarray:
    return await asyncio.to_thread(lambda: scipy.io.loadmat(path)["data"])


def _do_clustering_sync(
    data: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    out_root: str,
    scan_name: str,
    k_clusters: int,
    seed: int,
    device_str: str = None,
) -> Dict[str, Any]:
    """Run feature extraction, PCA, kmeans and save montages and xy map. Synchronous helper for executor."""
    import logging

    logger = logging.getLogger(f"analyze_scan_{scan_name}")

    set_seed(seed)
    import torch

    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"   🖥️ Using device: {device}")

    logger.info("   🔧 STEP 3: Feature extraction...")
    extractor = GPUExtractor(
        device, NBINS, HARMONICS, RAD_BANDS, CENTER_MASK_RADIUS, RMAX, NTHETA, BATCH
    )
    X = extractor.extract(data)
    logger.info(f"      ✅ Extracted features: {X.shape}")

    logger.info("   📊 STEP 4: Principal Component Analysis (PCA)...")
    Xt = torch.from_numpy(X).to(device)
    n_components = min(50, Xt.shape[1])
    Z, _, _ = pca_torch(Xt, n_components=n_components)
    logger.info(f"      ✅ PCA completed: reduced to {n_components} components")

    logger.info("   🎯 STEP 5: K-means clustering...")
    labels_t, min_d2_t, _ = kmeans_torch(
        Z, K=k_clusters, iters=50, restarts=2, seed=seed
    )
    labels_np = labels_t.cpu().numpy().astype(np.int32)
    min_d2_np = min_d2_t.cpu().numpy().astype(np.float32)
    logger.info(f"      ✅ K-means completed: {k_clusters} clusters identified")

    logger.info("   🖼️ STEP 6: Generating visualization outputs...")
    out_dir = Path(out_root) / scan_name
    montage_dir = out_dir / "montages"

    # Save cluster montages with 4x4 grid (16 pictures)
    save_cluster_montages(
        data, labels_np, min_d2_np, montage_dir, grid=CLUSTER_MONTAGE_GRID
    )
    logger.info(f"      📁 Saved cluster montages to: {montage_dir}")

    # Save individual images for each cluster
    individual_images_dir = out_dir / "individual_images"
    save_individual_cluster_images(
        data, labels_np, min_d2_np, individual_images_dir, images_per_cluster=16
    )
    logger.info(f"      📁 Saved individual cluster images to: {individual_images_dir}")

    # Map clusters to semantic keys (cycle through available keys)
    semantic_keys = CATEGORIES
    user_map = {i: semantic_keys[i % len(semantic_keys)] for i in range(k_clusters)}

    out_dir.mkdir(parents=True, exist_ok=True)
    final_xy_path = out_dir / f"{scan_name}_xy_map_FINAL.png"
    xy_plot_final(
        xs,
        ys,
        labels_np,
        user_map,
        get_color_map(user_map),
        final_xy_path,
        title=f"{scan_name} K={k_clusters}",
    )
    logger.info(f"      🗺️ Saved classification map to: {final_xy_path}")

    npz_path = out_dir / f"{scan_name}_cluster_results.npz"
    np.savez(npz_path, labels=labels_np, xs=xs, ys=ys)
    logger.info(f"      💾 Saved cluster data to: {npz_path}")

    return {
        "out_dir": str(out_dir),
        "montage_dir": str(montage_dir),
        "individual_images_dir": str(individual_images_dir),
        "xy_map": str(final_xy_path),
        "npz": str(npz_path),
        "k": int(k_clusters),
    }


async def analyze_scan(
    sql_driver,
    scan_identifier: Union[int, str],
    out_root: str,
    k_clusters: int = 16,
    seed: int = 0,
    device: str = None,
) -> Dict[str, Any]:
    """
    Analyze a scan stored in the database using the MCP server's sql_driver.

    Args:
        sql_driver: MCP sql driver (async interface) obtained from server.get_sql_driver().
        scan_identifier: scan id (int) or scan_name (str).
        out_root: output root folder for montages and maps.
        k_clusters: number of clusters for kmeans.
        seed: random seed.
        device: torch device string or None to auto-detect.

    Returns:
        dict with paths to generated outputs and cluster assignment count.
    """
    import logging

    # Get a logger that matches the one from the main tool
    logger = logging.getLogger(f"analyze_scan_{scan_identifier}")

    logger.info("🔍 STEP 2: Starting data loading from database...")

    # build query depending on identifier type
    if isinstance(scan_identifier, int):
        condition_column = "s.id"
        param = scan_identifier
        scan_id = scan_identifier
    else:
        condition_column = "s.scan_name"
        param = str(scan_identifier)
        # Get scan_id for database operations
        scan_query = "SELECT id FROM scans WHERE scan_name = %s"
        scan_result = await sql_driver.execute_query(scan_query, [param])
        if not scan_result:
            raise RuntimeError(f"Could not find scan with name: {scan_identifier}")
        scan_id = scan_result[0].cells["id"]

    logger.info(f"   📋 Scan ID resolved: {scan_id}")

    query = f"""
        SELECT
            rmf.id as mat_file_id,
            rmf.row_index,
            rmf.file_path
        FROM raw_mat_files rmf
        JOIN scans s ON rmf.scan_id = s.id
        WHERE {condition_column} = {{}}
        ORDER BY rmf.row_index ASC;
    """

    rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [param])

    if not rows:
        raise RuntimeError(f"Could not find scan with identifier: {scan_identifier}")

    logger.info(f"   📁 Found {len(rows)} .mat files to process")

    # collect files and coordinates, keep track of mat_file_id for database updates
    file_entries: List[Dict[str, Any]] = [r.cells for r in rows]
    data_chunks = []
    xs = []
    ys = []
    mat_file_mapping = []  # Track which patterns come from which mat files

    # load .mat files concurrently in threads
    logger.info("   💾 Loading .mat files...")
    load_tasks = []
    for entry in file_entries:
        mat_file_id = int(entry["mat_file_id"])
        row_index = (
            int(entry["row_index"])
            if "row_index" in entry
            else int(entry.get("row", 0))
        )
        file_path = entry["file_path"]
        load_tasks.append((mat_file_id, row_index, file_path))

    files_processed = 0
    for mat_file_id, row_index, file_path in load_tasks:
        if not os.path.isfile(file_path):
            # skip missing
            logger.warning(f"     ⚠️ Skipping missing file: {file_path}")
            continue
        mat = await _load_mat_async(file_path)
        mat_data = np.asarray(mat, dtype=np.float32)
        num_cols = mat_data.shape[0]
        data_chunks.append(mat_data)
        xs.extend([int(row_index)] * num_cols)
        ys.extend(list(range(1, num_cols + 1)))
        # Track mat_file_id and col_index for each pattern
        for col_idx in range(1, num_cols + 1):
            mat_file_mapping.append((mat_file_id, col_idx))

        files_processed += 1
        if files_processed % 10 == 0 or files_processed == len(load_tasks):
            logger.info(f"     📊 Processed {files_processed}/{len(load_tasks)} files")

    if not data_chunks:
        raise RuntimeError("No valid .mat data loaded for this scan.")

    data = np.vstack(data_chunks)
    xs = np.array(xs, dtype=np.int32)
    ys = np.array(ys, dtype=np.int32)

    logger.info(
        f"   ✅ Data loading complete: {data.shape[0]} diffraction patterns loaded"
    )
    logger.info(f"      Shape: {data.shape}, Data type: {data.dtype}")

    # run heavy clustering in executor to avoid blocking event loop
    logger.info(
        "🔄 STEP 3-6: Running clustering analysis (this may take several minutes)..."
    )
    logger.info(
        f"   🎯 Parameters: k_clusters={k_clusters}, seed={seed}, device={device}"
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        _do_clustering_sync,
        data,
        xs,
        ys,
        out_root,
        str(param),
        k_clusters,
        seed,
        device,
    )

    logger.info("   ✅ Clustering analysis completed")
    logger.info(f"   📊 Generated {result['k']} clusters")
    logger.info(f"   📁 Output directory: {result['out_dir']}")

    # Store cluster assignments in database
    logger.info("📋 Starting database storage of cluster assignments...")
    labels = np.load(result["npz"])["labels"]
    logger.info(f"   💾 Loading cluster labels: {len(labels)} assignments")

    # Create clustering run record
    cluster_run_query = """
        INSERT INTO clustering_runs (scan_id, k_value, algorithm_details, results_path)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """
    algorithm_details = (
        f"K-means clustering with seed={seed}, device={device or 'auto'}"
    )
    cluster_run_result = await sql_driver.execute_query(
        cluster_run_query, [scan_id, k_clusters, algorithm_details, result["out_dir"]]
    )
    clustering_run_id = cluster_run_result[0].cells["id"]
    logger.info(f"   🗄️ Created clustering run record: ID {clustering_run_id}")

    # Create identified_clusters records
    cluster_counts = {}
    for label in labels:
        cluster_counts[int(label)] = cluster_counts.get(int(label), 0) + 1

    logger.info(f"   🏷️ Creating cluster records for {len(cluster_counts)} clusters...")
    cluster_ids = {}
    for cluster_index, pattern_count in cluster_counts.items():
        cluster_query = """
            INSERT INTO identified_clusters (run_id, cluster_index, pattern_count)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        cluster_result = await sql_driver.execute_query(
            cluster_query, [clustering_run_id, cluster_index, pattern_count]
        )
        cluster_ids[cluster_index] = cluster_result[0].cells["id"]

    logger.info("   ✅ Cluster records created")

    # Update diffraction_patterns with cluster assignments
    logger.info("   💾 Updating diffraction patterns with cluster assignments...")
    updated_patterns = 0

    batch_size = 1000  # Process in batches for better progress tracking
    total_patterns = len(mat_file_mapping)

    for i, (mat_file_id, col_index) in enumerate(mat_file_mapping):
        cluster_label = int(labels[i])

        # First, ensure diffraction_pattern record exists
        pattern_insert_query = """
            INSERT INTO diffraction_patterns (source_mat_id, col_index, cluster_label, clustering_run_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_mat_id, col_index) 
            DO UPDATE SET 
                cluster_label = EXCLUDED.cluster_label,
                clustering_run_id = EXCLUDED.clustering_run_id
        """
        await sql_driver.execute_query(
            pattern_insert_query,
            [mat_file_id, col_index, cluster_label, clustering_run_id],
        )

        # Get the pattern_id for pattern_cluster_assignments
        pattern_id_query = """
            SELECT id FROM diffraction_patterns 
            WHERE source_mat_id = %s AND col_index = %s
        """
        pattern_result = await sql_driver.execute_query(
            pattern_id_query, [mat_file_id, col_index]
        )

        if pattern_result:
            pattern_id = pattern_result[0].cells["id"]
            cluster_db_id = cluster_ids[cluster_label]

            # Insert into pattern_cluster_assignments
            assignment_query = """
                INSERT INTO pattern_cluster_assignments (pattern_id, cluster_id)
                VALUES (%s, %s)
                ON CONFLICT (pattern_id, cluster_id) DO NOTHING
            """
            await sql_driver.execute_query(
                assignment_query, [pattern_id, cluster_db_id]
            )
            updated_patterns += 1

        # Progress logging
        if (i + 1) % batch_size == 0 or (i + 1) == total_patterns:
            progress_pct = ((i + 1) / total_patterns) * 1000
            logger.info(
                f"     📊 Database update progress: {i + 1}/{total_patterns} ({progress_pct:.1f}%)"
            )

    # Update scan record with classification map path
    scan_update_query = """
        UPDATE scans 
        SET classification_map_path = %s 
        WHERE id = %s
    """
    await sql_driver.execute_query(scan_update_query, [result["xy_map"], scan_id])
    logger.info("   🗺️ Updated scan record with classification map path")

    result["clustering_run_id"] = clustering_run_id
    result["updated_patterns"] = updated_patterns
    result["total_patterns"] = len(labels)
    result["classification_map_saved"] = True

    logger.info(
        f"✅ Database storage completed: {updated_patterns}/{len(labels)} patterns saved"
    )

    return result


async def regenerate_classification_map(
    sql_driver,
    scan_identifier: Union[int, str],
    out_path: Union[str, Path],
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
        "SELECT rmf.row_index, dp.col_index, dp.cluster_label "
        "FROM diffraction_patterns dp "
        "JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id "
        "WHERE dp.clustering_run_id = %s "
        "ORDER BY rmf.row_index, dp.col_index"
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
