#!/usr/bin/env python3
"""
Simple script to test database connection and list available scans.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool

async def list_scans(config_path=None):
    """List all available scans in the database."""
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
        
        # Query for scans using the connection pool
        pool = db_connection.pool
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT id, scan_name, folder_path, created_at
                    FROM scans
                    ORDER BY created_at DESC;
                """)
                rows = await cursor.fetchall()
        
        if not rows:
            print("No scans found in the database.")
        else:
            print(f"\nFound {len(rows)} scan(s):")
            print("-" * 80)
            print(f"{'ID':<5} {'Name':<15} {'Folder Path':<30} {'Created At':<20}")
            print("-" * 80)
            for row in rows:
                print(f"{row[0]:<5} {row[1]:<15} {row[2]:<30} {row[3]!s:<20}")
        
        # Close the connection pool
        await db_connection.close()
        print("\nDatabase connection closed.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(list_scans())
