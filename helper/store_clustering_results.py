import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postgres_mcp.analyze.db_analyze import draw_cluster_classification_map, store_clustering_results
from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool, SqlDriver
from postgres_mcp.workflows import data_root


async def store_clustering_results_cli(scan_identifier, clustering_run_id, npz_file_path, config_path=None):
    pool = DbConnPool()
    folder = None
    saved = False
    try:
        await pool.pool_connect(ConfigManager(config_path).get_database_url())
        driver = SqlDriver(conn=pool)
        column = 'id' if isinstance(scan_identifier, int) else 'scan_name'
        rows = await driver.execute_query(f'SELECT id FROM scans WHERE {column} = %s', [scan_identifier])
        if not rows:
            raise ValueError('Scan not found')
        with np.load(npz_file_path) as data:
            labels, xs, ys = data['labels'], data['xs'], data['ys']
        if not len(labels):
            raise ValueError('Cluster arrays are empty')
        k = int(labels.max()) + 1
        folder = data_root() / 'results' / uuid4().hex
        folder.mkdir(parents=True)
        target = folder / 'clustering.npz'
        shutil.copy2(npz_file_path, target)
        map_path = folder / 'cluster_map.png'
        await asyncio.to_thread(draw_cluster_classification_map, xs, ys, labels, k, map_path)
        parameters = {'source': 'external_npz', 'source_run_id': clustering_run_id, 'source_file': str(Path(npz_file_path).resolve())}
        (folder / 'parameters.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
        result = await store_clustering_results(driver, rows[0].cells['id'],
                    {'npz': str(target), 'out_dir': str(folder), 'xy_map': str(map_path), 'k': k}, parameters)
        saved = True
        return result
    finally:
        await pool.close()
        if folder and not saved:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import external cluster arrays as a new analysis run')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scan-id', type=int)
    selection.add_argument('--scan-name')
    parser.add_argument('--clustering-run-id', type=int, help='Source run ID, retained as provenance')
    parser.add_argument('--npz-file', required=True)
    parser.add_argument('--config')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(store_clustering_results_cli(
        args.scan_id if args.scan_id is not None else args.scan_name, args.clustering_run_id, args.npz_file, args.config)), indent=2))
