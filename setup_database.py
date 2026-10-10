import argparse
import asyncio

from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool
from postgres_mcp.workflows import initialize_schema


async def init_database(config_path=None):
    pool = DbConnPool()
    try:
        await pool.pool_connect(ConfigManager(config_path).get_database_url())
        await initialize_schema(pool.pool)
        print('Database schema is ready. Existing scans and results were preserved.')
    finally:
        await pool.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create or upgrade the 4DLLM database without deleting data')
    parser.add_argument('--config')
    args = parser.parse_args()
    asyncio.run(init_database(args.config))
