#!/usr/bin/env python3
"""
Standalone CLI tool for analyzing 4D-STEM scans without MCP/LLM connection.

This tool directly accesses the database and performs the same analysis as the
analyze_scan_tool but without the overhead of the MCP server, which should
help identify performance bottlenecks and avoid connection timeouts.
"""

import argparse
import asyncio
import logging
import os
import time
import sys
from pathlib import Path
from typing import Union, Dict, Any

# Set matplotlib backend before any imports that might use it
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add the project root to the path so we can import postgres_mcp modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from postgres_mcp.analyze.db_analyze import analyze_scan
from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool, SqlDriver
from postgres_mcp.lock_manager import LockManager
from postgres_mcp.background_logger import setup_background_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def analyze_scan_cli(
    scan_identifier: Union[int, str],
    out_root: str = "/tmp/scan_analysis",
    k_clusters: int = 16,
    seed: int = 0,
    device: str = None,
    config_path: str = None,
    job_id: str = None,
) -> Dict[str, Any]:
    """
    Analyze a 4D-STEM scan directly without MCP server.

    Args:
        scan_identifier: scan id (int) or scan_name (str)
        out_root: output root folder for analysis results
        k_clusters: number of clusters for k-means clustering
        seed: random seed for reproducibility
        device: torch device string or None to auto-detect
        config_path: path to database config file
        job_id: unique identifier for this job

    Returns:
        dict with paths to generated outputs
    """
    # Set up background logging
    bg_logger, log_file = setup_background_logging(job_id or f"job-{scan_identifier}")
    bg_logger.info(f"Starting analysis for scan: {scan_identifier}")
    bg_logger.info(f"Log file: {log_file}")
    
    # Flush stdout/stderr to ensure immediate output
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        bg_logger.info("Loading database configuration...")
        
        # Load database configuration
        config_manager = ConfigManager(
            config_path=config_path or "config/database.json"
        )
        database_url = config_manager.get_database_url()

        if not database_url:
            raise ValueError("Failed to construct database URL from configuration.")

        bg_logger.info("Connecting to database...")

        # Initialize database connection pool
        db_connection = DbConnPool()
        await db_connection.pool_connect(database_url)

        bg_logger.info("Connected to database successfully.")

        # Create SQL driver
        sql_driver = SqlDriver(conn=db_connection)

        # Run the analysis
        bg_logger.info("Starting scan analysis...")
        bg_logger.info("This may take several minutes depending on scan size...")
        sys.stdout.flush()
        sys.stderr.flush()
        
        result = await analyze_scan(
            sql_driver=sql_driver,
            scan_identifier=scan_identifier,
            out_root=out_root,
            k_clusters=k_clusters,
            seed=seed,
            device=device,
        )

        # Close the connection pool
        await db_connection.close()
        bg_logger.info("Database connection closed.")
        sys.stdout.flush()
        sys.stderr.flush()

        return result

    except Exception as e:
        bg_logger.error(f"Error analyzing scan '{scan_identifier}': {e}", exc_info=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 4D-STEM scans directly without MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scan-id 1
  %(prog)s --scan-name "sample_scan" --out-root "./results" --k-clusters 8
  %(prog)s --scan-id 2 --device "cpu" --config "./config/db.json"
        """,
    )

    # Required arguments
    scan_group = parser.add_mutually_exclusive_group(required=True)
    scan_group.add_argument("--scan-id", type=int, help="Scan ID (integer) to analyze")
    scan_group.add_argument(
        "--scan-name", type=str, help="Scan name (string) to analyze"
    )

    # Optional arguments
    parser.add_argument(
        "--out-root",
        type=str,
        default="/tmp/scan_analysis",
        help="Output root directory for analysis results (default: /tmp/scan_analysis)",
    )

    parser.add_argument(
        "--k-clusters",
        type=int,
        default=16,
        help="Number of clusters for k-means clustering (default: 16)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cpu', 'cuda', etc.) or None to auto-detect (default: None)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to database configuration JSON file (default: config/database.json)",
    )

    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Unique job identifier (default: auto-generated)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    
    

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine scan identifier
    scan_identifier = args.scan_id if args.scan_id is not None else args.scan_name

    # Validate output directory
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    # Create job info for lock management
    job_id = args.job_id or f"job-{scan_identifier}-{int(time.time())}"
    
    # Set up background logging immediately
    bg_logger, log_file = setup_background_logging(job_id)
    bg_logger.info(f"Analysis CLI started for scan: {scan_identifier}")
    bg_logger.info(f"Log file: {log_file}")
    bg_logger.info(f"Output directory: {out_root}")
    print(f"Logging to: {log_file}")  # Immediate output for user
    
    # Flush stdout to ensure immediate visibility
    sys.stdout.flush()
    sys.stderr.flush()

    # Create job info for lock management
    job_info = {
        "job_id": job_id,
        "scan_identifier": str(scan_identifier),
        "pid": os.getpid(),
        "timestamp": time.time(),
    }

    # Initialize lock manager
    lock_manager = LockManager()

    # Try to acquire lock
    if not lock_manager.acquire_lock(job_info):
        # Check if lock is stale
        if lock_manager.check_and_clean_stale_lock():
            # Try to acquire lock again after cleaning stale lock
            if not lock_manager.acquire_lock(job_info):
                lock_info = lock_manager.get_lock_info()
                error_msg = f"System is busy. Another analysis task is running: {lock_info}"
                bg_logger.error(error_msg)
                print(f"Error: {error_msg}")
                sys.exit(1)
        else:
            lock_info = lock_manager.get_lock_info()
            error_msg = f"System is busy. Another analysis task is running: {lock_info}"
            bg_logger.error(error_msg)
            print(f"Error: {error_msg}")
            sys.exit(1)

    # Ensure lock is released when the script exits
    try:
        # Run the analysis
        bg_logger.info("Starting standalone scan analysis...")
        result = asyncio.run(
            analyze_scan_cli(
                scan_identifier=scan_identifier,
                out_root=out_root,
                k_clusters=args.k_clusters,
                seed=args.seed,
                device=args.device,
                config_path=args.config,
                job_id=job_id,
            )
        )

        bg_logger.info("Analysis completed successfully!")
        print("\nAnalysis Results:")
        print("-" * 50)
        for key, value in result.items():
            print(f"{key:15}: {value}")
            bg_logger.info(f"Result - {key}: {value}")

        # Print summary
        print(f"\nResults saved to: {result.get('out_dir', 'unknown')}")
        print(f"XY map: {result.get('xy_map', 'unknown')}")
        print(f"Montages: {result.get('montage_dir', 'unknown')}")
        
        bg_logger.info(f"Results saved to: {result.get('out_dir', 'unknown')}")
        bg_logger.info(f"XY map: {result.get('xy_map', 'unknown')}")
        bg_logger.info(f"Montages: {result.get('montage_dir', 'unknown')}")

    except Exception as e:
        bg_logger.error(f"Analysis failed: {e}")
        print(f"Error: Analysis failed - {e}")
        sys.exit(1)
    finally:
        # Always release the lock
        lock_manager.release_lock()
        bg_logger.info("Lock released")
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
