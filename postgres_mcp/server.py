import argparse
import asyncio
import datetime
import logging
import os
import signal
import sys
import shutil
import scipy.io
import subprocess
import time
import uuid
import torch
from enum import Enum
from typing import Any
from typing import List
from typing import Union
from .config import ConfigManager

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .sql import DbConnPool
from .sql import SafeSqlDriver
from .sql import SqlDriver
from .sql import obfuscate_password
from .analyze import analyze_scan
from .import_4dstem import process_one_mib
from .lock_manager import LockManager
from .cif_analysis import CIFManager, PatternSimulator, PatternComparator

# Initialize FastMCP with default settings
mcp = FastMCP("postgres-mcp")

# Constants
PG_STAT_STATEMENTS = "pg_stat_statements"
HYPOPG_EXTENSION = "hypopg"

ResponseType = List[types.TextContent | types.ImageContent | types.EmbeddedResource]

logger = logging.getLogger(__name__)


class AccessMode(str, Enum):
    """SQL access modes for the server."""

    UNRESTRICTED = "unrestricted"  # Unrestricted access
    RESTRICTED = "restricted"  # Read-only with safety features


db_connection = DbConnPool()
current_access_mode = AccessMode.UNRESTRICTED
shutdown_in_progress = False

# 创建长期存在的Driver实例
_unrestricted_driver = SqlDriver(conn=db_connection)
_restricted_driver = SafeSqlDriver(sql_driver=_unrestricted_driver, timeout=30)


async def get_sql_driver() -> Union[SqlDriver, SafeSqlDriver]:
    """Get the appropriate SQL driver based on the current access mode."""
    # 不再创建新实例，而是根据模式返回预先创建好的实例
    if current_access_mode == AccessMode.RESTRICTED:
        logger.debug("Using pre-configured SafeSqlDriver (RESTRICTED mode)")
        return _restricted_driver
    else:
        logger.debug("Using pre-configured unrestricted SqlDriver (UNRESTRICTED mode)")
        return _unrestricted_driver


def format_text_response(text: Any) -> ResponseType:
    """Format a text response."""
    return [types.TextContent(type="text", text=str(text))]


def format_error_response(error: str) -> ResponseType:
    """Format an error response."""
    return format_text_response(f"Error: {error}")


@mcp.tool(description="List all schemas in the database")
async def list_schemas() -> ResponseType:
    """List all schemas in the database."""
    try:
        sql_driver = await get_sql_driver()
        rows = await sql_driver.execute_query(
            """
            SELECT
                schema_name,
                schema_owner,
                CASE
                    WHEN schema_name LIKE 'pg_%' THEN 'System Schema'
                    WHEN schema_name = 'information_schema' THEN 'System Information Schema'
                    ELSE 'User Schema'
                END as schema_type
            FROM information_schema.schemata
            ORDER BY schema_type, schema_name
            """
        )
        schemas = [row.cells for row in rows] if rows else []
        return format_text_response(schemas)
    except Exception as e:
        logger.error(f"Error listing schemas: {e}")
        return format_error_response(str(e))


@mcp.tool(description="List objects in a schema")
async def list_objects(
    schema_name: str = Field(description="Schema name"),
    object_type: str = Field(
        description="Object type: 'table', 'view', 'sequence', or 'extension'",
        default="table",
    ),
) -> ResponseType:
    """List objects of a given type in a schema."""
    try:
        sql_driver = await get_sql_driver()

        if object_type in ("table", "view"):
            table_type = "BASE TABLE" if object_type == "table" else "VIEW"
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = {} AND table_type = {}
                ORDER BY table_name
                """,
                [schema_name, table_type],
            )
            objects = (
                [
                    {
                        "schema": row.cells["table_schema"],
                        "name": row.cells["table_name"],
                        "type": row.cells["table_type"],
                    }
                    for row in rows
                ]
                if rows
                else []
            )

        elif object_type == "sequence":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT sequence_schema, sequence_name, data_type
                FROM information_schema.sequences
                WHERE sequence_schema = {}
                ORDER BY sequence_name
                """,
                [schema_name],
            )
            objects = (
                [
                    {
                        "schema": row.cells["sequence_schema"],
                        "name": row.cells["sequence_name"],
                        "data_type": row.cells["data_type"],
                    }
                    for row in rows
                ]
                if rows
                else []
            )

        elif object_type == "extension":
            # Extensions are not schema-specific
            rows = await sql_driver.execute_query(
                """
                SELECT extname, extversion, extrelocatable
                FROM pg_extension
                ORDER BY extname
                """
            )
            objects = (
                [
                    {
                        "name": row.cells["extname"],
                        "version": row.cells["extversion"],
                        "relocatable": row.cells["extrelocatable"],
                    }
                    for row in rows
                ]
                if rows
                else []
            )

        else:
            return format_error_response(f"Unsupported object type: {object_type}")

        return format_text_response(objects)
    except Exception as e:
        logger.error(f"Error listing objects: {e}")
        return format_error_response(str(e))


@mcp.tool(description="Show detailed information about a database object")
async def get_object_details(
    schema_name: str = Field(description="Schema name"),
    object_name: str = Field(description="Object name"),
    object_type: str = Field(
        description="Object type: 'table', 'view', 'sequence', or 'extension'",
        default="table",
    ),
) -> ResponseType:
    """Get detailed information about a database object."""
    try:
        sql_driver = await get_sql_driver()

        if object_type in ("table", "view"):
            # Get columns
            col_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = {} AND table_name = {}
                ORDER BY ordinal_position
                """,
                [schema_name, object_name],
            )
            columns = (
                [
                    {
                        "column": r.cells["column_name"],
                        "data_type": r.cells["data_type"],
                        "is_nullable": r.cells["is_nullable"],
                        "default": r.cells["column_default"],
                    }
                    for r in col_rows
                ]
                if col_rows
                else []
            )

            # Get constraints
            con_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT tc.constraint_name, tc.constraint_type, kcu.column_name
                FROM information_schema.table_constraints AS tc
                LEFT JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = {} AND tc.table_name = {}
                """,
                [schema_name, object_name],
            )

            constraints = {}
            if con_rows:
                for row in con_rows:
                    cname = row.cells["constraint_name"]
                    ctype = row.cells["constraint_type"]
                    col = row.cells["column_name"]

                    if cname not in constraints:
                        constraints[cname] = {"type": ctype, "columns": []}
                    if col:
                        constraints[cname]["columns"].append(col)

            constraints_list = [
                {"name": name, **data} for name, data in constraints.items()
            ]

            # Get indexes
            idx_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = {} AND tablename = {}
                """,
                [schema_name, object_name],
            )

            indexes = (
                [
                    {"name": r.cells["indexname"], "definition": r.cells["indexdef"]}
                    for r in idx_rows
                ]
                if idx_rows
                else []
            )

            result = {
                "basic": {
                    "schema": schema_name,
                    "name": object_name,
                    "type": object_type,
                },
                "columns": columns,
                "constraints": constraints_list,
                "indexes": indexes,
            }

        elif object_type == "sequence":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT sequence_schema, sequence_name, data_type, start_value, increment
                FROM information_schema.sequences
                WHERE sequence_schema = {} AND sequence_name = {}
                """,
                [schema_name, object_name],
            )

            if rows and rows[0]:
                row = rows[0]
                result = {
                    "schema": row.cells["sequence_schema"],
                    "name": row.cells["sequence_name"],
                    "data_type": row.cells["data_type"],
                    "start_value": row.cells["start_value"],
                    "increment": row.cells["increment"],
                }
            else:
                result = {}

        elif object_type == "extension":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT extname, extversion, extrelocatable
                FROM pg_extension
                WHERE extname = {}
                """,
                [object_name],
            )

            if rows and rows[0]:
                row = rows[0]
                result = {
                    "name": row.cells["extname"],
                    "version": row.cells["extversion"],
                    "relocatable": row.cells["extrelocatable"],
                }
            else:
                result = {}

        else:
            return format_error_response(f"Unsupported object type: {object_type}")

        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error getting object details: {e}")
        return format_error_response(str(e))


# @mcp.tool(description="Explains the execution plan for a SQL query, showing how the database will execute it and provides detailed cost estimates.")
# async def explain_query(
#     sql: str = Field(description="SQL query to explain"),
#     analyze: bool = Field(
#         description="When True, actually runs the query to show real execution statistics instead of estimates. "
#         "Takes longer but provides more accurate information.",
#         default=False,
#     ),
#     hypothetical_indexes: list[dict[str, Any]] = Field(
#         description="""A list of hypothetical indexes to simulate. Each index must be a dictionary with these keys:
#     - 'table': The table name to add the index to (e.g., 'users')
#     - 'columns': List of column names to include in the index (e.g., ['email'] or ['last_name', 'first_name'])
#     - 'using': Optional index method (default: 'btree', other options include 'hash', 'gist', etc.)

# Examples: [
#     {"table": "users", "columns": ["email"], "using": "btree"},
#     {"table": "orders", "columns": ["user_id", "created_at"]}
# ]
# If there is no hypothetical index, you can pass an empty list.""",
#         default=[],
#     ),
# ) -> ResponseType:
#     """
#     Explains the execution plan for a SQL query.

#     Args:
#         sql: The SQL query to explain
#         analyze: When True, actually runs the query for real statistics
#         hypothetical_indexes: Optional list of indexes to simulate
#     """
#     try:
#         sql_driver = await get_sql_driver()
#         explain_tool = ExplainPlanTool(sql_driver=sql_driver)
#         result: ExplainPlanArtifact | ErrorResult | None = None

#         # If hypothetical indexes are specified, check for HypoPG extension
#         if hypothetical_indexes and len(hypothetical_indexes) > 0:
#             if analyze:
#                 return format_error_response("Cannot use analyze and hypothetical indexes together")
#             try:
#                 # Use the common utility function to check if hypopg is installed
#                 (
#                     is_hypopg_installed,
#                     hypopg_message,
#                 ) = await check_hypopg_installation_status(sql_driver)

#                 # If hypopg is not installed, return the message
#                 if not is_hypopg_installed:
#                     return format_text_response(hypopg_message)

#                 # HypoPG is installed, proceed with explaining with hypothetical indexes
#                 result = await explain_tool.explain_with_hypothetical_indexes(sql, hypothetical_indexes)
#             except Exception:
#                 raise  # Re-raise the original exception
#         elif analyze:
#             try:
#                 # Use EXPLAIN ANALYZE
#                 result = await explain_tool.explain_analyze(sql)
#             except Exception:
#                 raise  # Re-raise the original exception
#         else:
#             try:
#                 # Use basic EXPLAIN
#                 result = await explain_tool.explain(sql)
#             except Exception:
#                 raise  # Re-raise the original exception

#         if result and isinstance(result, ExplainPlanArtifact):
#             return format_text_response(result.to_text())
#         else:
#             error_message = "Error processing explain plan"
#             if isinstance(result, ErrorResult):
#                 error_message = result.to_text()
#             return format_error_response(error_message)
#     except Exception as e:
#         logger.error(f"Error explaining query: {e}")
#         return format_error_response(str(e))


# Query function declaration without the decorator - we'll add it dynamically based on access mode
async def execute_sql(
    sql: str = Field(description="SQL to run", default="all"),
) -> ResponseType:
    """Executes a SQL query against the database."""
    try:
        sql_driver = await get_sql_driver()
        rows = await sql_driver.execute_query(sql)  # type: ignore
        if rows is None:
            return format_text_response("No results")
        return format_text_response(list([r.cells for r in rows]))
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return format_error_response(str(e))


# @mcp.tool(description="Analyze frequently executed queries in the database and recommend optimal indexes")
# @validate_call
# async def analyze_workload_indexes(
#     max_index_size_mb: int = Field(description="Max index size in MB", default=10000),
#     method: Literal["dta", "llm"] = Field(description="Method to use for analysis", default="dta"),
# ) -> ResponseType:
#     """Analyze frequently executed queries in the database and recommend optimal indexes."""
#     try:
#         sql_driver = await get_sql_driver()
#         if method == "dta":
#             index_tuning = DatabaseTuningAdvisor(sql_driver)
#         else:
#             index_tuning = LLMOptimizerTool(sql_driver)
#         dta_tool = TextPresentation(sql_driver, index_tuning)
#         result = await dta_tool.analyze_workload(max_index_size_mb=max_index_size_mb)
#         return format_text_response(result)
#     except Exception as e:
#         logger.error(f"Error analyzing workload: {e}")
#         return format_error_response(str(e))


# @mcp.tool(description="Analyze a list of (up to 10) SQL queries and recommend optimal indexes")
# @validate_call
# async def analyze_query_indexes(
#     queries: list[str] = Field(description="List of Query strings to analyze"),
#     max_index_size_mb: int = Field(description="Max index size in MB", default=10000),
#     method: Literal["dta", "llm"] = Field(description="Method to use for analysis", default="dta"),
# ) -> ResponseType:
#     """Analyze a list of SQL queries and recommend optimal indexes."""
#     if len(queries) == 0:
#         return format_error_response("Please provide a non-empty list of queries to analyze.")
#     if len(queries) > MAX_NUM_INDEX_TUNING_QUERIES:
#         return format_error_response(f"Please provide a list of up to {MAX_NUM_INDEX_TUNING_QUERIES} queries to analyze.")

#     try:
#         sql_driver = await get_sql_driver()
#         if method == "dta":
#             index_tuning = DatabaseTuningAdvisor(sql_driver)
#         else:
#             index_tuning = LLMOptimizerTool(sql_driver)
#         dta_tool = TextPresentation(sql_driver, index_tuning)
#         result = await dta_tool.analyze_queries(queries=queries, max_index_size_mb=max_index_size_mb)
#         return format_text_response(result)
#     except Exception as e:
#         logger.error(f"Error analyzing queries: {e}")
#         return format_error_response(str(e))


# @mcp.tool(
#     description="Analyzes database health. Here are the available health checks:\n"
#     "- index - checks for invalid, duplicate, and bloated indexes\n"
#     "- connection - checks the number of connection and their utilization\n"
#     "- vacuum - checks vacuum health for transaction id wraparound\n"
#     "- sequence - checks sequences at risk of exceeding their maximum value\n"
#     "- replication - checks replication health including lag and slots\n"
#     "- buffer - checks for buffer cache hit rates for indexes and tables\n"
#     "- constraint - checks for invalid constraints\n"
#     "- all - runs all checks\n"
#     "You can optionally specify a single health check or a comma-separated list of health checks. The default is 'all' checks."
# )
# async def analyze_db_health(
#     health_type: str = Field(
#         description=f"Optional. Valid values are: {', '.join(sorted([t.value for t in HealthType]))}.",
#         default="all",
#     ),
# ) -> ResponseType:
#     """Analyze database health for specified components.

#     Args:
#         health_type: Comma-separated list of health check types to perform.
#                     Valid values: index, connection, vacuum, sequence, replication, buffer, constraint, all
#     """
#     health_tool = DatabaseHealthTool(await get_sql_driver())
#     result = await health_tool.health(health_type=health_type)
#     return format_text_response(result)


# @mcp.tool(
#     name="get_top_queries",
#     description=f"Reports the slowest or most resource-intensive queries using data from the '{PG_STAT_STATEMENTS}' extension.",
# )
# async def get_top_queries(
#     sort_by: str = Field(
#         description="Ranking criteria: 'total_time' for total execution time or 'mean_time' for mean execution time per call, or 'resources' "
#         "for resource-intensive queries",
#         default="resources",
#     ),
#     limit: int = Field(description="Number of queries to return when ranking based on mean_time or total_time", default=10),
# ) -> ResponseType:
#     try:
#         sql_driver = await get_sql_driver()
#         top_queries_tool = TopQueriesCalc(sql_driver=sql_driver)

#         if sort_by == "resources":
#             result = await top_queries_tool.get_top_resource_queries()
#             return format_text_response(result)
#         elif sort_by == "mean_time" or sort_by == "total_time":
#             # Map the sort_by values to what get_top_queries_by_time expects
#             result = await top_queries_tool.get_top_queries_by_time(limit=limit, sort_by="mean" if sort_by == "mean_time" else "total")
#         else:
#             return format_error_response("Invalid sort criteria. Please use 'resources' or 'mean_time' or 'total_time'.")
#         return format_text_response(result)
#     except Exception as e:
#         logger.error(f"Error getting slow queries: {e}")
#         return format_error_response(str(e))


@mcp.tool(
    description="Run clustering analysis on a 4D-STEM scan stored in the database. This performs feature extraction, PCA, k-means clustering, and generates montages and XY maps."
)
async def analyze_scan_tool(
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan to analyze."
    ),
    out_root: str = Field(
        description="Output root directory where analysis results will be saved.",
        default="/tmp/scan_analysis",
    ),
    k_clusters: int = Field(
        description="Number of clusters for k-means clustering.", default=16
    ),
) -> ResponseType:
    """
    Analyze a 4D-STEM scan using clustering and generate visualization outputs.

    This tool:
    1. Loads .mat files for the specified scan from the database
    2. Extracts features using GPU-accelerated polar coordinate analysis
    3. Performs PCA dimensionality reduction
    4. Runs k-means clustering
    5. Generates cluster montages and XY maps
    6. Saves results to the specified output directory

    Returns paths to generated outputs including montages, XY maps, and cluster data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sql_driver = await get_sql_driver()
        result = await analyze_scan(
            sql_driver=sql_driver,
            scan_identifier=scan_identifier,
            out_root=out_root,
            k_clusters=k_clusters,
            seed=0,
            device=device,
        )
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error analyzing scan '{scan_identifier}': {e}", exc_info=True)
        return format_error_response(str(e))


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a custom database config JSON file.",
        default="config/database.json",
    )
    # parser.add_argument(
    #     "--access-mode",
    #     type=str,
    #     choices=[mode.value for mode in AccessMode],
    #     default=AccessMode.UNRESTRICTED.value,
    #     help="Set SQL access mode: unrestricted (unrestricted) or restricted (read-only with protections)",
    # )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Select MCP transport: stdio (default) or sse",
    )
    parser.add_argument(
        "--sse-host",
        type=str,
        default="localhost",
        help="Host to bind SSE server to (default: localhost)",
    )
    parser.add_argument(
        "--sse-port",
        type=int,
        default=8000,
        help="Port for SSE server (default: 8000)",
    )

    args = parser.parse_args()
    args.transport = "stdio"  # Force STDIO transport for async and background support

    # Store the access mode in the global variable
    global current_access_mode
    current_access_mode = AccessMode.UNRESTRICTED

    # Add the query tool with a description appropriate to the access mode
    if current_access_mode == AccessMode.UNRESTRICTED:
        mcp.add_tool(execute_sql, description="Execute any SQL query")
    else:
        mcp.add_tool(execute_sql, description="Execute a read-only SQL query")

    logger.info(f"Starting PostgreSQL MCP Server in {current_access_mode.upper()} mode")

    try:
        config_manager = ConfigManager(config_path=args.config)
        database_url = config_manager.get_database_url()
        if not database_url:
            raise ValueError("Failed to construct database URL from configuration.")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)  # 配置错误，直接退出

    if not database_url:
        raise ValueError(
            "Error: No database URL provided. Please specify via 'DATABASE_URI' environment variable or command-line argument.",
        )

    # Initialize database connection pool
    try:
        await db_connection.pool_connect(database_url)
        logger.info(
            "Successfully connected to database and initialized connection pool"
        )
    except Exception as e:
        logger.warning(
            f"Could not connect to database: {obfuscate_password(str(e))}",
        )
        logger.warning(
            "The MCP server will start but database operations will fail until a valid connection is established.",
        )

    # Set up proper shutdown handling
    try:
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        logger.warning("Signal handling not supported on Windows")
        pass

    # Run the server with the selected transport (always async)
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        # Update FastMCP settings based on command line arguments
        mcp.settings.host = args.sse_host
        mcp.settings.port = args.sse_port
        await mcp.run_sse_async()


async def shutdown(sig=None):
    """Clean shutdown of the server."""
    global shutdown_in_progress

    if shutdown_in_progress:
        logger.warning("Forcing immediate exit")
        sys.exit(1)

    shutdown_in_progress = True

    if sig:
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = str(sig)
        logger.info(f"Received exit signal {sig_name}")

    # Close database connections
    try:
        await db_connection.close()
        logger.info("Closed database connections")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    # Determine exit code: 128 + signal number
    exit_code = (
        128 + (sig.value if isinstance(sig, signal.Signals) else sig)
        if sig is not None
        else 0
    )
    sys.exit(exit_code)


@mcp.tool(
    description="Processes a .mib file, unpacks it into .mat files, and catalogs the entire scan structure into the database."
)
async def ingest_scan_from_mib(
    source_mib_path: str = Field(
        description="The absolute path to the source .mib file on the server's filesystem."
    ),
) -> ResponseType:
    """
    When you run this function, you are running the first step of the 4D-STEM data ingestion pipeline. You are a helpful assistant that helps users process and ingest new 4D-STEM scans from .mib files into a PostgreSQL database.
    This is a one-stop tool to process and ingest a new 4D-STEM scan from a .mib file.
    This tool performs a multi-step workflow:
    1. Standardizes file locations by copying the source .mib to a managed 'Raw' directory.
    2. Unpacks the .mib file into a structured folder of .mat files within a 'Data' directory.
    3. Catalogs the entire file structure (scan, .mat files, diffraction points) into the PostgreSQL database.
    """
    # --- 0. 标准化目录和数据库配置 ---
    # 使用 os.path.dirname(__file__) 来动态确定当前文件位置，构建相对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(
        os.path.join(base_dir, "..")
    )  # 假设 server.py 在 postgres_mcp/ 目录下
    raw_data_dir = os.path.join(project_root, "Raw")
    processed_data_dir = os.path.join(project_root, "Data")

    logger.info(f"Project Root detected at: {project_root}")
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # 获取 SQL 驱动 (需要使用无限制的驱动来进行数据插入)
    sql_driver = _unrestricted_driver

    try:
        # --- 阶段 0: 准备工作 (文件标准化) ---
        logger.info("Phase 0: Standardizing file locations.")
        if not os.path.exists(source_mib_path):
            raise FileNotFoundError(f"Source file not found: {source_mib_path}")

        mib_filename = os.path.basename(source_mib_path)
        managed_mib_path = os.path.join(raw_data_dir, mib_filename)
        shutil.copy2(source_mib_path, managed_mib_path)
        logger.info(f"Copied source .mib to managed location: {managed_mib_path}")

        # --- 阶段 1: 解包 .mib 文件 ---
        logger.info(f"Phase 1: Unpacking {mib_filename} to .mat directory.")
        scan_name = os.path.splitext(mib_filename)[0]
        mat_folder_path = os.path.join(processed_data_dir, scan_name)

        # 调用您的处理函数
        process_one_mib(managed_mib_path, processed_data_dir)

        if not os.path.exists(mat_folder_path):
            raise IOError(f".mat folder was not generated at {mat_folder_path}")
        logger.info(f"Unpacking complete. Data is in: {mat_folder_path}")

        # --- 阶段 2: 将文件结构编目到数据库 ---
        logger.info("Phase 2: Cataloging file structure into PostgreSQL.")

        # 检查扫描是否已存在
        existing_scan = await SafeSqlDriver.execute_param_query(
            sql_driver,
            "SELECT id FROM scans WHERE scan_name = {};",
            [scan_name],
        )
        if existing_scan:
            return format_error_response(
                f"Scan '{scan_name}' already exists in the database. Please rename the .mib file or clean the database manually."
            )

        # 1. 插入 scan 记录并获取ID
        scan_result = await sql_driver.execute_query(
            "INSERT INTO scans (scan_name, folder_path) VALUES (%s, %s) RETURNING id;",
            [scan_name, mat_folder_path],
        )
        if not scan_result or len(scan_result) == 0:
            raise RuntimeError("Failed to insert scan record")

        scan_id = scan_result[0].cells["id"]
        logger.info(f"Created new scan record with id: {scan_id}")

        # 2. 遍历 .mat 文件并批量插入
        mat_files = sorted(
            [f for f in os.listdir(mat_folder_path) if f.endswith(".mat")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        total_patterns = 0
        for mat_file in mat_files:
            row_index = int(os.path.splitext(mat_file)[0])
            file_path = os.path.join(mat_folder_path, mat_file)

            # 插入 raw_mat_file 记录
            mat_result = await sql_driver.execute_query(
                "INSERT INTO raw_mat_files (scan_id, row_index, file_path) VALUES (%s, %s, %s) RETURNING id;",
                [scan_id, row_index, file_path],
            )
            if not mat_result or len(mat_result) == 0:
                raise RuntimeError(f"Failed to insert mat file record for {mat_file}")

            mat_id = mat_result[0].cells["id"]

            # 读取 .mat 文件获取列数，然后批量插入 diffraction_patterns
            mat_data = scipy.io.loadmat(file_path)["data"]
            num_cols = mat_data.shape[0]

            # 批量插入 diffraction_patterns
            for col_idx in range(num_cols):
                await sql_driver.execute_query(
                    "INSERT INTO diffraction_patterns (source_mat_id, col_index) VALUES (%s, %s);",
                    [mat_id, col_idx + 1],
                )

            logger.info(
                f"  ↳ Cataloged {mat_file} (mat_id={mat_id}): {num_cols} diffraction patterns."
            )
            total_patterns += num_cols

        success_message = (
            f"Successfully ingested scan '{scan_name}'.\n"
            f"- Scan ID: {scan_id}\n"
            f"- Total .mat files processed: {len(mat_files)}\n"
            f"- Total diffraction patterns cataloged: {total_patterns}"
        )
        logger.info(success_message)
        return format_text_response(success_message)

    except Exception as e:
        logger.error(f"Error during ingestion process: {e}", exc_info=True)
        return format_error_response(str(e))


@mcp.tool(
    description="Lists a high-level summary of all scientific scans ingested from .mib files."
)
async def list_ingested_scans() -> ResponseType:
    """
    Retrieves a top-level summary of all scans stored in the database.

    This tool provides the most essential information for each scan: its unique ID,
    the scan name (derived from the original .mib file), its original folder path,
    and the date it was cataloged. It deliberately avoids details about .mat files
    or diffraction patterns to keep the overview clean and fast.
    """
    try:
        sql_driver = await get_sql_driver()

        # 【已修改】查询语句现在变得非常简单，只查询 scans 表
        query = """
            SELECT 
                id AS scan_id,
                scan_name,
                folder_path,
                created_at AS ingestion_date
            FROM 
                scans
            ORDER BY 
                ingestion_date DESC;
        """

        rows = await sql_driver.execute_query(query)

        if not rows:
            return format_text_response(
                "No scans have been ingested into the database yet."
            )

        scans_summary = [row.cells for row in rows]

        return format_text_response(scans_summary)

    except Exception as e:
        logger.error(f"Error listing ingested scans: {e}", exc_info=True)
        return format_error_response(str(e))


# Add missing decorator so it’s registered as a tool
@mcp.tool(
    description="Retrieves a detailed list of raw .mat files for a scan by its ID or name."
)
async def get_scan_details(
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) associated to one scan in detail."
    ),
) -> ResponseType:
    """
    Retrieves a detailed list of all raw .mat files associated with a single scan.

    You can identify the scan either by its integer `scan_id` (preferred for accuracy)
    or its string `scan_name`. The tool returns the row index and the full file path
    for each .mat file belonging to that scan.
    """
    try:
        sql_driver = await get_sql_driver()

        # 根据输入的标识符类型，动态构建查询条件
        if isinstance(scan_identifier, int):
            # 通过 ID 查询，更精确
            condition_column = "s.id"
            param = scan_identifier
        else:
            # 通过名称查询
            condition_column = "s.scan_name"
            param = str(scan_identifier)

        query = f"""
            SELECT 
                rmf.row_index,
                rmf.file_path,
                rmf.id AS mat_file_id
            FROM 
                raw_mat_files rmf
            JOIN 
                scans s ON rmf.scan_id = s.id
            WHERE 
                {condition_column} = {{}} 
            ORDER BY 
                rmf.row_index ASC;
        """

        # 使用参数化查询来防止SQL注入
        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [param])

        if not rows:
            return format_error_response(
                f"Could not find a scan with identifier: '{scan_identifier}'. Please check the ID or name."
            )

        mat_files_list = [row.cells for row in rows]

        # 为了提供更丰富的上下文，我们同时返回扫描的基本信息
        scan_info_query = f"SELECT id, scan_name, folder_path FROM scans WHERE {condition_column} = {{}};"
        scan_info_rows = await SafeSqlDriver.execute_param_query(
            sql_driver, scan_info_query, [param]
        )
        scan_info = scan_info_rows[0].cells if scan_info_rows else {}

        response_payload = {"scan_info": scan_info, "mat_files": mat_files_list}

        return format_text_response(response_payload)

    except Exception as e:
        logger.error(
            f"Error getting scan details for '{scan_identifier}': {e}", exc_info=True
        )
        return format_error_response(str(e))


@mcp.tool(description="Check the status of background analysis jobs.")
@mcp.tool(description="Check the status of background analysis jobs.")
async def check_analysis_status() -> ResponseType:
    """
    Check the status of background analysis jobs.

    Returns information about currently running analysis tasks or indicates
    that no analysis is currently running.
    """
    try:
        if LockManager.is_locked():
            lock_info = LockManager.get_lock_info()
            if lock_info:
                # Check if the process is still running
                pid = lock_info.get("pid")
                job_id = lock_info.get("job_id", "unknown")
                scan_identifier = lock_info.get("scan_identifier", "unknown")
                timestamp = lock_info.get("timestamp", 0)

                # Format the start time
                import datetime

                start_time = datetime.datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Check log file for progress
                import os

                log_dir = "/tmp/4dllm_logs"
                log_file = os.path.join(log_dir, f"analysis_{job_id}.log")
                progress_info = ""

                if os.path.exists(log_file):
                    # Read last few lines of the log file for progress
                    try:
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            # Get last 10 lines
                            last_lines = lines[-10:] if len(lines) > 10 else lines
                            progress_info = "\nRecent log entries:\n" + "".join(
                                last_lines
                            )
                    except Exception as e:
                        progress_info = f"\nCould not read log file: {e}"

                status_msg = (
                    f"Analysis job is currently running:\n"
                    f"- Job ID: {job_id}\n"
                    f"- Scan: {scan_identifier}\n"
                    f"- Started: {start_time}\n"
                    f"- Process PID: {pid}\n"
                    f"- Log file: {log_file}"
                    f"{progress_info}"
                )

                return format_text_response(status_msg)
            else:
                return format_text_response(
                    "System is locked, but no job information available."
                )
        else:
            return format_text_response("No analysis jobs are currently running.")

    except Exception as e:
        logger.error(f"Error checking analysis status: {e}", exc_info=True)
        return format_error_response(str(e))


@mcp.tool(
    description="Download a CIF file from a crystallography database and store it in the database"
)
async def download_cif_file(
    url: str = Field(description="The URL to download the CIF file from"),
) -> ResponseType:
    """
    Downloads a CIF file from a crystallography database and stores it in the database.
    """
    try:
        # Get SQL driver and create CIF manager
        sql_driver = _unrestricted_driver  # Need unrestricted for INSERT operations
        cif_manager = CIFManager(sql_driver)

        # Download and process the CIF file
        file_path = await cif_manager.download_cif(url)
        cif_info = cif_manager.parse_cif(file_path)
        cif_id = await cif_manager.store_cif_info(
            os.path.basename(file_path), file_path, cif_info
        )

        return format_text_response(
            f"Successfully downloaded and stored CIF file.\n"
            f"- Database ID: {cif_id}\n"
            f"- Filename: {os.path.basename(file_path)}\n"
            f"- Crystal system: {cif_info.get('crystal_system', 'Unknown')}\n"
            f"- Space group: {cif_info.get('space_group', 'Unknown')}"
        )

    except Exception as e:
        logger.error(f"Failed to download CIF file: {e}")
        return format_error_response(f"Failed to download CIF file: {str(e)}")


@mcp.tool(description="Upload a local CIF file and store it in the database")
async def upload_cif_file(
    file_path: str = Field(description="The absolute path to the local CIF file"),
) -> ResponseType:
    """
    Uploads a local CIF file and stores it in the database.
    """
    try:
        # Get SQL driver and create CIF manager
        sql_driver = _unrestricted_driver  # Need unrestricted for INSERT operations
        cif_manager = CIFManager(sql_driver)

        # Upload and process the CIF file
        dest_path = await cif_manager.upload_cif(file_path)
        cif_info = cif_manager.parse_cif(dest_path)
        cif_id = await cif_manager.store_cif_info(
            os.path.basename(dest_path), dest_path, cif_info
        )

        return format_text_response(
            f"Successfully uploaded and stored CIF file.\n"
            f"- Database ID: {cif_id}\n"
            f"- Filename: {os.path.basename(dest_path)}\n"
            f"- Crystal system: {cif_info.get('crystal_system', 'Unknown')}\n"
            f"- Space group: {cif_info.get('space_group', 'Unknown')}"
        )

    except Exception as e:
        logger.error(f"Failed to upload CIF file: {e}")
        return format_error_response(f"Failed to upload CIF file: {str(e)}")


@mcp.tool(description="Generate simulated diffraction patterns from a CIF file")
async def generate_simulated_patterns(
    cif_id: int = Field(description="The database ID of the CIF file"),
    count: int = Field(
        description="Number of simulated patterns to generate", default=100
    ),
) -> ResponseType:
    """
    Generates simulated diffraction patterns from a CIF file.
    """
    try:
        # Get SQL driver and create pattern simulator
        sql_driver = _unrestricted_driver
        simulator = PatternSimulator(sql_driver)

        # Generate simulated patterns with correct parameter passing
        result = await simulator.generate_patterns(cif_id, count=count)

        return format_text_response(f"Simulation result for CIF ID {cif_id}: {result}")
    except Exception as e:
        logger.error(f"Failed to generate simulated patterns: {e}")
        return format_error_response(f"Failed to generate simulated patterns: {str(e)}")


@mcp.tool(
    description="Compare experimental diffraction patterns with simulated patterns from a CIF file"
)
async def compare_patterns(
    scan_id: int = Field(description="The database ID of the scan to compare"),
    cif_id: int = Field(description="The database ID of the CIF file to compare with"),
) -> ResponseType:
    """
    Compares experimental diffraction patterns with simulated patterns from a CIF file.
    """
    try:
        # Get SQL driver and create pattern comparator
        sql_driver = _unrestricted_driver
        comparator = PatternComparator(sql_driver)

        # Perform batch comparison
        results = await comparator.batch_compare(scan_id, cif_id)

        return format_text_response(
            f"Comparison result for scan {scan_id} vs CIF {cif_id}: {results}"
        )
    except Exception as e:
        logger.error(f"Failed to compare patterns: {e}")
        return format_error_response(f"Failed to compare patterns: {str(e)}")


@mcp.tool(description="List all CIF files in the database")
async def list_cif_files() -> ResponseType:
    """
    Lists all CIF files stored in the database.
    """
    try:
        # Get SQL driver and create CIF manager
        sql_driver = await get_sql_driver()
        cif_manager = CIFManager(sql_driver)

        # Get list of CIF files
        cif_files = await cif_manager.list_cif_files()

        # Format the results
        if not cif_files:
            return format_text_response("No CIF files found in the database.")

        result_lines = ["CIF Files in Database:", "=" * 50]
        for cif_file in cif_files:
            result_lines.extend(
                [
                    f"ID: {cif_file['id']} - {cif_file['filename']}",
                    f"  Crystal System: {cif_file['crystal_system'] or 'Unknown'}",
                    f"  Space Group: {cif_file['space_group'] or 'Unknown'}",
                    f"  Uploaded: {cif_file['uploaded_at']}",
                    "",
                ]
            )

        return format_text_response("\n".join(result_lines))

    except Exception as e:
        logger.error(f"Failed to list CIF files: {e}")
        return format_error_response(f"Failed to list CIF files: {str(e)}")


if __name__ == "__main__":
    # This is the entry point when running the server directly
    asyncio.run(main())
