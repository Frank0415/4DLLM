import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postgres_mcp.analyze.db_analyze import analyze_scan
from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool, SqlDriver


async def analyze_scan_cli(scan_identifier, out_root=None, k_clusters=16, seed=0, device=None, config_path=None, job_id=None):
    pool = DbConnPool()
    try:
        await pool.pool_connect(ConfigManager(config_path).get_database_url())
        return await analyze_scan(SqlDriver(conn=pool), scan_identifier, out_root, k_clusters, seed, device)
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze an imported 4D-STEM scan')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scan-id', type=int)
    selection.add_argument('--scan-name')
    parser.add_argument('--out-root')
    parser.add_argument('--k-clusters', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--config')
    parser.add_argument('--job-id')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr)
    result = asyncio.run(analyze_scan_cli(args.scan_id if args.scan_id is not None else args.scan_name,
                         args.out_root, args.k_clusters, args.seed, args.device, args.config, args.job_id))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
