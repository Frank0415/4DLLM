#!/usr/bin/env python3
"""
Database initialization script for 4D-STEM research project.
This script sets up the PostgreSQL database with all required tables.
"""

import asyncio
import logging
import os
import sys
from typing import Optional

# Add the project root to the path so we can import postgres_mcp modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from postgres_mcp.config import ConfigManager
from postgres_mcp.sql import DbConnPool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_database(config_path: Optional[str] = None) -> None:
    """Initialize the database with required tables."""
    try:
        # Load database configuration
        config_manager = ConfigManager(config_path=config_path or "config/database.json")
        database_url = config_manager.get_database_url()
        
        if not database_url:
            raise ValueError("Failed to construct database URL from configuration.")
        
        logger.info("Connecting to database...")
        
        # Initialize database connection pool
        db_connection = DbConnPool()
        await db_connection.pool_connect(database_url)
        
        logger.info("Connected to database successfully.")
        
        # Read SQL schema
        schema_path = os.path.join(os.path.dirname(__file__), "SQL", "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        
        # Get a connection and create tables
        async with db_connection.pool.connection() as conn:
            await conn.execute(schema_sql)
        
        logger.info("Database tables created successfully.")
        
        # Close the connection pool
        await db_connection.close()
        logger.info("Database connection closed.")
        
        print("\n" + "="*60)
        print("DATABASE INITIALIZATION COMPLETE")
        print("="*60)
        print("The database has been initialized with support for:")
        print("  • Raw 4D-STEM data storage")
        print("  • K-Means clustering workflows")
        print("  • LLM-powered pattern analysis")
        print("  • Spatial pattern distribution queries")
        print("  • Cluster statistics and analytics")
        print("  • CIF file management and pattern simulation")
        print("  • Pattern comparison between experimental and simulated data")
        print("\nTables created: 14")
        print("Views created: 7")
        print("Indexes created: 30+")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    # Run the async function
    asyncio.run(init_database())