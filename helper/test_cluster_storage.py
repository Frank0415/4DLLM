#!/usr/bin/env python3
"""
Test script to verify cluster result storage functionality.
"""

import asyncio
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool

async def test_cluster_storage(config_path=None):
    """Test the cluster result storage functionality."""
    try:
        # Load database configuration
        config_manager = ConfigManager(config_path=config_path or "config/database.json")
        database_url = config_manager.get_database_url()
        
        if not database_url:
            print("Failed to construct database URL from configuration.")
            return
        
        print("Connecting to database...")
        
        # Initialize database connection pool
        db_connection = DbConnPool()
        await db_connection.pool_connect(database_url)
        
        print("Connected to database successfully.")
        
        # Create test data (simulating clustering results)
        # For a small 10x10 scan area
        test_size = 100  # 10x10 = 100 patterns
        labels = np.random.randint(0, 5, test_size)  # 5 clusters (0-4)
        xs = np.repeat(np.arange(10), 10)  # x coordinates 0-9 repeated 10 times
        ys = np.tile(np.arange(10), 10)    # y coordinates 0-9 tiled 10 times
        
        print(f"Generated test data: {test_size} patterns with {len(np.unique(labels))} clusters")
        
        # Save test data to NPZ file
        test_npz_path = "/tmp/test_clustering_results.npz"
        np.savez(test_npz_path, labels=labels, xs=xs, ys=ys)
        print(f"Saved test data to: {test_npz_path}")
        
        # Show sample of the data
        print("\nSample of test data:")
        print("Pattern | X | Y | Cluster")
        print("--------|---|---|--------")
        for i in range(min(10, len(labels))):
            print(f"{i:7} | {xs[i]:1} | {ys[i]:1} | {labels[i]:1}")
        
        # Close the connection pool
        await db_connection.close()
        print("\nDatabase connection closed.")
        
        print(f"\nTest completed successfully!")
        print(f"Test NPZ file created at: {test_npz_path}")
        print(f"You can now use this file with the store_clustering_results.py tool")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cluster_storage())
