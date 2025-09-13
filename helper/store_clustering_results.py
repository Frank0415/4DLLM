#!/usr/bin/env python3
"""
Standalone CLI tool for storing k-means clustering results in the database.

This tool directly accesses the database and stores clustering results without
the overhead of the MCP server, which should help avoid connection timeouts.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Union

# Set matplotlib backend before any imports that might use it
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add the project root to the path so we can import postgres_mcp modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool, SafeSqlDriver
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def store_clustering_results_cli(
    scan_identifier: Union[int, str],
    clustering_run_id: int,
    npz_file_path: str,
    config_path: str = None,
) -> dict:
    """
    Store k-means clustering results directly in the database.
    
    Args:
        scan_identifier: Scan ID (int) or name (str)
        clustering_run_id: ID of the clustering run
        npz_file_path: Path to .npz file with clustering results
        config_path: Path to database config file
        
    Returns:
        dict with statistics about stored results
    """
    try:
        logger.info(f"Loading clustering results from: {npz_file_path}")
        
        # Load clustering results from NPZ file
        with np.load(npz_file_path) as data:
            labels = data['labels']
            xs = data['xs'] 
            ys = data['ys']
        
        logger.info(f"Loaded {len(labels)} cluster labels from NPZ file")
        
        # Load database configuration
        config_manager = ConfigManager(
            config_path=config_path or "config/database.json"
        )
        database_url = config_manager.get_database_url()

        if not database_url:
            raise ValueError("Failed to construct database URL from configuration.")

        logger.info("Connecting to database...")

        # Initialize database connection pool
        db_connection = DbConnPool()
        await db_connection.pool_connect(database_url)

        logger.info("Connected to database successfully.")

        # Get SQL driver
        sql_driver = db_connection

        # Determine scan ID
        if isinstance(scan_identifier, int):
            scan_id = scan_identifier
        else:
            # Get scan ID from name
            scan_query = "SELECT id FROM scans WHERE scan_name = %s"
            async with db_connection.pool.connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(scan_query, [str(scan_identifier)])
                    scan_rows = await cursor.fetchall()
            if not scan_rows:
                raise ValueError(f"Scan '{scan_identifier}' not found")
            scan_id = scan_rows[0][0]

        logger.info(f"Processing scan ID: {scan_id}")

        # Update database with cluster labels
        updated_count = 0
        error_count = 0
        
        # Process in batches for better performance
        batch_size = 1000
        logger.info(f"Updating database with cluster labels in batches of {batch_size}")
        
        for i in range(0, len(labels), batch_size):
            batch_end = min(i + batch_size, len(labels))
            logger.info(f"Processing batch {i//batch_size + 1}: patterns {i} to {batch_end}")
            
            # Process batch
            for j in range(i, batch_end):
                label = labels[j]
                x = xs[j]
                y = ys[j]
                
                try:
                    # Update the diffraction pattern with its cluster label
                    update_query = """
                        UPDATE diffraction_patterns 
                        SET cluster_label = %s, clustering_run_id = %s
                        FROM raw_mat_files 
                        WHERE diffraction_patterns.source_mat_id = raw_mat_files.id 
                        AND raw_mat_files.scan_id = %s 
                        AND raw_mat_files.row_index = %s 
                        AND diffraction_patterns.col_index = %s
                    """
                    async with db_connection.pool.connection() as conn:
                        async with conn.cursor() as cursor:
                            await cursor.execute(
                                update_query, 
                                [int(label), clustering_run_id, scan_id, int(x), int(y)]
                            )
                    updated_count += 1
                    
                    # Show progress every 1000 updates
                    if updated_count % 1000 == 0:
                        logger.info(f"Updated {updated_count} patterns so far...")
                        
                except Exception as e:
                    logger.warning(f"Failed to update pattern at ({x}, {y}): {e}")
                    error_count += 1

        # Create clustering run record if it doesn't exist
        try:
            run_check_query = "SELECT id FROM clustering_runs WHERE id = %s"
            async with db_connection.pool.connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(run_check_query, [clustering_run_id])
                    run_exists = await cursor.fetchall()
                    
            if not run_exists:
                # Insert clustering run record
                insert_run_query = """
                    INSERT INTO clustering_runs (id, scan_id, k_value, algorithm_details)
                    VALUES (%s, %s, %s, %s)
                """
                k_value = len(np.unique(labels))
                algo_details = f"K-means clustering with k={k_value}"
                async with db_connection.pool.connection() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(
                            insert_run_query,
                            [clustering_run_id, scan_id, k_value, algo_details]
                        )
                logger.info(f"Created new clustering run record with ID: {clustering_run_id}")
        except Exception as e:
            logger.warning(f"Failed to create clustering run record: {e}")

        # Close the connection pool
        await db_connection.close()
        logger.info("Database connection closed.")

        result_stats = {
            "updated_patterns": updated_count,
            "errors": error_count,
            "clustering_run_id": clustering_run_id,
            "unique_clusters": len(np.unique(labels)),
            "total_patterns": len(labels)
        }
        
        logger.info(f"Storage completed successfully: {result_stats}")
        return result_stats

    except Exception as e:
        logger.error(f"Error storing clustering results: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Store k-means clustering results in the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scan-id 1 --clustering-run-id 100 --npz-file "/path/to/results.npz"
  %(prog)s --scan-name "sample_scan" --clustering-run-id 101 --npz-file "./cluster_results.npz" --config "./config/db.json"
        """,
    )

    # Required arguments
    parser.add_argument(
        "--scan-id", 
        type=int, 
        help="Scan ID (integer) to associate with results"
    )
    
    parser.add_argument(
        "--scan-name", 
        type=str, 
        help="Scan name (string) to associate with results"
    )
    
    parser.add_argument(
        "--clustering-run-id",
        type=int,
        required=True,
        help="ID of the clustering run to associate with these results"
    )
    
    parser.add_argument(
        "--npz-file",
        type=str,
        required=True,
        help="Path to the .npz file containing clustering results (labels, xs, ys)"
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to database configuration JSON file (default: config/database.json)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.scan_id and not args.scan_name:
        print("Error: Either --scan-id or --scan-name must be specified")
        sys.exit(1)
        
    if args.scan_id and args.scan_name:
        print("Error: Only one of --scan-id or --scan-name should be specified")
        sys.exit(1)
        
    scan_identifier = args.scan_id if args.scan_id is not None else args.scan_name

    # Validate NPZ file exists
    if not os.path.exists(args.npz_file):
        print(f"Error: NPZ file not found: {args.npz_file}")
        sys.exit(1)
        
    # Validate NPZ file contents
    try:
        with np.load(args.npz_file) as data:
            required_keys = ['labels', 'xs', 'ys']
            for key in required_keys:
                if key not in data:
                    print(f"Error: Required key '{key}' not found in NPZ file")
                    sys.exit(1)
    except Exception as e:
        print(f"Error: Invalid NPZ file format: {e}")
        sys.exit(1)

    # Run the storage process
    try:
        logger.info("Starting clustering results storage process...")
        result = asyncio.run(
            store_clustering_results_cli(
                scan_identifier=scan_identifier,
                clustering_run_id=args.clustering_run_id,
                npz_file_path=args.npz_file,
                config_path=args.config,
            )
        )

        logger.info("Storage completed successfully!")
        print("\nStorage Results:")
        print("-" * 50)
        print(f"Total patterns processed: {result['total_patterns']}")
        print(f"Patterns updated in DB:     {result['updated_patterns']}")
        print(f"Errors encountered:         {result['errors']}")
        print(f"Clustering run ID:          {result['clustering_run_id']}")
        print(f"Unique clusters found:     {result['unique_clusters']}")
        print(f"\nQuery examples:")
        print(f"  SELECT * FROM diffraction_patterns WHERE clustering_run_id = {result['clustering_run_id']} AND cluster_label = 3;")
        print(f"  SELECT cluster_label, COUNT(*) FROM diffraction_patterns WHERE clustering_run_id = {result['clustering_run_id']} GROUP BY cluster_label;")

    except Exception as e:
        logger.error(f"Storage process failed: {e}")
        print(f"Error: Storage process failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()