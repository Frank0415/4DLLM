import os
import asyncio
from pathlib import Path
from typing import Union, Dict, Any, List

import numpy as np
import scipy.io

# reuse clustering and plotting utilities from pre_class_v2
from class_base import (
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
    MONTAGE_GRID,
    FINAL_COLOR_MAP,
)

# NOTE: sql_driver is expected to be an object provided by the MCP server (SqlDriver or SafeSqlDriver)
# which exposes async methods: execute_param_query(query, params) -> list[Row], where Row.cells contains column dict.


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
    set_seed(seed)
    import torch

    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")

    extractor = GPUExtractor(
        device, NBINS, HARMONICS, RAD_BANDS, CENTER_MASK_RADIUS, RMAX, NTHETA, BATCH
    )
    X = extractor.extract(data)

    Xt = torch.from_numpy(X).to(device)
    Z, _, _ = pca_torch(Xt, n_components=min(50, Xt.shape[1]))
    labels_t, min_d2_t, _ = kmeans_torch(
        Z, K=k_clusters, iters=50, restarts=2, seed=seed
    )
    labels_np = labels_t.cpu().numpy().astype(np.int32)
    min_d2_np = min_d2_t.cpu().numpy().astype(np.float32)

    out_dir = Path(out_root) / scan_name
    montage_dir = out_dir / "montages"
    save_cluster_montages(data, labels_np, min_d2_np, montage_dir, grid=MONTAGE_GRID)

    # Map clusters to semantic keys (cycle through available keys)
    semantic_keys = list(FINAL_COLOR_MAP.keys())
    user_map = {i: semantic_keys[i % len(semantic_keys)] for i in range(k_clusters)}

    out_dir.mkdir(parents=True, exist_ok=True)
    final_xy_path = out_dir / f"{scan_name}_xy_map_FINAL.png"
    xy_plot_final(
        xs,
        ys,
        labels_np,
        user_map,
        FINAL_COLOR_MAP,
        final_xy_path,
        title=f"{scan_name} K={k_clusters}",
    )

    npz_path = out_dir / f"{scan_name}_cluster_results.npz"
    np.savez(npz_path, labels=labels_np, xs=xs, ys=ys)

    return {
        "out_dir": str(out_dir),
        "montage_dir": str(montage_dir),
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
        dict with paths to generated outputs.
    """
    # build query depending on identifier type
    if isinstance(scan_identifier, int):
        condition_column = "s.id"
        param = scan_identifier
    else:
        condition_column = "s.scan_name"
        param = str(scan_identifier)

    query = f"""
        SELECT
            rmf.row_index,
            rmf.file_path
        FROM raw_mat_files rmf
        JOIN scans s ON rmf.scan_id = s.id
        WHERE {condition_column} = %s
        ORDER BY rmf.row_index ASC;
    """

    rows = await sql_driver.execute_param_query(query, [param])
    if not rows:
        raise RuntimeError(f"Could not find scan with identifier: {scan_identifier}")

    # collect files and coordinates
    file_entries: List[Dict[str, Any]] = [r.cells for r in rows]
    data_chunks = []
    xs = []
    ys = []

    # load .mat files concurrently in threads
    load_tasks = []
    for entry in file_entries:
        row_index = (
            int(entry["row_index"])
            if "row_index" in entry
            else int(entry.get("row", 0))
        )
        file_path = entry["file_path"]
        load_tasks.append((row_index, file_path))

    for row_index, file_path in load_tasks:
        if not os.path.isfile(file_path):
            # skip missing
            continue
        mat = await _load_mat_async(file_path)
        mat_data = np.asarray(mat, dtype=np.float32)
        num_cols = mat_data.shape[0]
        data_chunks.append(mat_data)
        xs.extend([int(row_index)] * num_cols)
        ys.extend(list(range(1, num_cols + 1)))

    if not data_chunks:
        raise RuntimeError("No valid .mat data loaded for this scan.")

    data = np.vstack(data_chunks)
    xs = np.array(xs, dtype=np.int32)
    ys = np.array(ys, dtype=np.int32)

    # run heavy clustering in executor to avoid blocking event loop
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
    return result
