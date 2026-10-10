import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from postgres_mcp.analyze.db_analyze import analyze_scan
from postgres_mcp.config import ConfigManager
from postgres_mcp.llm_analysis import AnalysisPipeline
from postgres_mcp.sql import DbConnPool, SqlDriver
from postgres_mcp.workflows import ingest_scan
from .jobs import JobCancelled, JobProgress, now, read_job, write_job


async def run(path):
    job = read_job(path)
    progress = JobProgress(path)
    pool = DbConnPool()
    job.update(status='running', pid=os.getpid(), updated_at=now())
    write_job(path, job)
    try:
        progress('starting', 0, 'Starting task')
        await pool.pool_connect(ConfigManager().get_database_url())
        driver = SqlDriver(conn=pool)
        params = job['parameters']
        if job['kind'] == 'import':
            result = await ingest_scan(driver, params['path'], params.get('scan_shape'), params.get('scan_name'), progress)
        elif job['kind'] == 'cluster':
            result = await analyze_scan(driver, params['scan_id'], k_clusters=params['k'], seed=params['seed'],
                                        device=params.get('device'), progress=progress)
        elif job['kind'] == 'llm':
            config = ConfigManager().get_api_keys()
            if not config or not config.get('api_keys') or any('your-api-key' in key for key in config['api_keys']):
                raise ValueError('Configure LLM API keys in Settings')
            rows = await driver.execute_query('SELECT scan_name FROM scans WHERE id = %s', [params['scan_id']])
            if not rows:
                raise ValueError('Scan not found')
            pipeline = AnalysisPipeline(driver, config['api_keys'], config['base_url'], config['model'])
            result = await pipeline.analyze_scan_clusters(rows[0].cells['scan_name'], params.get('cluster_id'),
                        params.get('samples', 5), params.get('concurrency', 3), progress, clustering_run_id=params.get('run_id'))
            if result['status'] == 'error':
                raise ValueError(result['message'])
        else:
            raise ValueError(f"Unknown task: {job['kind']}")
        job = read_job(path)
        job.update(status='succeeded', progress=1, stage='complete', message='Complete', result=result)
    except JobCancelled:
        job = read_job(path)
        job.update(status='cancelled', message='Cancelled')
    except Exception as error:
        logging.exception('Task failed')
        job = read_job(path)
        job.update(status='failed', error=str(error), message=str(error))
    finally:
        await pool.close()
        job['updated_at'] = now()
        write_job(path, job)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(Path(sys.argv[1])))
