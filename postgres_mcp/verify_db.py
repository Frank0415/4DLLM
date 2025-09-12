"""Script to verify database setup for 4D-STEM data."""
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

async def verify_database_setup(config_path: Optional[str] = None) -> None:
    """Verify that the required database tables exist."""
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
        
        # Check if required tables exist
        required_tables = ['scans', 'raw_mat_files', 'diffraction_patterns']
        
        for table in required_tables:
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """
            async with db_connection.pool.acquire() as conn:
                result = await conn.fetchval(query, table)
                if result:
                    logger.info(f"Table '{table}' exists.")
                else:
                    logger.warning(f"Table '{table}' does not exist.")
        
        # Close the connection pool
        await db_connection.close()
        logger.info("Database connection closed.")
        
    except Exception as e:
        logger.error(f"Failed to verify database setup: {e}")
        raise

if __name__ == "__main__":
    # Run the async function
    asyncio.run(verify_database_setup())