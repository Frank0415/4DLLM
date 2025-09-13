import os
import asyncio
from pathlib import Path
from typing import Union, Dict, Any, List

import numpy as np
import scipy.io

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
    MONTAGE_GRID,
    FINAL_COLOR_MAP,
)
from ..sql import SafeSqlDriver


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
    save_cluster_montages(data, labels_np, min_d2_np, montage_dir, grid=MONTAGE_GRID)
    logger.info(f"      📁 Saved cluster montages to: {montage_dir}")

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
    logger.info(f"      🗺️ Saved classification map to: {final_xy_path}")

    npz_path = out_dir / f"{scan_name}_cluster_results.npz"
    np.savez(npz_path, labels=labels_np, xs=xs, ys=ys)
    logger.info(f"      💾 Saved cluster data to: {npz_path}")

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
