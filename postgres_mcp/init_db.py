"""Database initialization script to create required tables for 4D-STEM data."""
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

# SQL statements to create the required tables
CREATE_TABLES_SQL = """
-- Table to store scan metadata
CREATE TABLE IF NOT EXISTS scans (
    id SERIAL PRIMARY KEY,
    scan_name VARCHAR(255) UNIQUE NOT NULL,
    folder_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store raw .mat files associated with each scan
CREATE TABLE IF NOT EXISTS raw_mat_files (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER REFERENCES scans(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scan_id, row_index)
);

-- Table to store diffraction patterns
CREATE TABLE IF NOT EXISTS diffraction_patterns (
    id SERIAL PRIMARY KEY,
    source_mat_id INTEGER REFERENCES raw_mat_files(id) ON DELETE CASCADE,
    col_index INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_mat_id, col_index)
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_raw_mat_files_scan_id ON raw_mat_files(scan_id);
CREATE INDEX IF NOT EXISTS idx_raw_mat_files_row_index ON raw_mat_files(row_index);
CREATE INDEX IF NOT EXISTS idx_diffraction_patterns_source_mat_id ON diffraction_patterns(source_mat_id);
CREATE INDEX IF NOT EXISTS idx_diffraction_patterns_col_index ON diffraction_patterns(col_index);
"""

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
        
        # Get a connection and create tables
        async with db_connection.pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
        
        logger.info("Database tables created successfully.")
        
        # Close the connection pool
        await db_connection.close()
        logger.info("Database connection closed.")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    # Run the async function
    asyncio.run(init_database())