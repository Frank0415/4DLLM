import argparse
import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import scipy.io
import torch
import numpy as np
from enum import Enum
from typing import Any
from typing import List
from typing import Union, Optional
from .config import ConfigManager
from PIL import Image as pil_Image
from .analyze.db_analyze import regenerate_classification_map

logger = logging.getLogger(__name__)

import mcp.types as types
from mcp.server.fastmcp import FastMCP, Context, Image
from pydantic import Field

from .sql import DbConnPool
from .sql import SafeSqlDriver
from .sql import SqlDriver
from .sql import obfuscate_password
from .analyze import analyze_scan
from .workflows import ingest_scan, data_root, initialize_schema
from .cif_analysis import CIFManager, PatternSimulator, PatternComparator
from .llm_analysis import AnalysisPipeline
from .llm_analysis.consensus_analyzer import ConsensusAnalyzer
from .llm_analysis.llm_orchestrator import LLMOrchestrator
from .mcp_images.mcp_image import fetch_images

# JSON Prompt Template from working implementation
JSON_PROMPT_TEMPLATE = """
You are an expert materials scientist specializing in the analysis of 4D-STEM and electron diffraction data. Your primary task is to meticulously analyze the provided 2D image, which is a single convergent beam electron diffraction (CBED) pattern, and classify the material's state at the probed location.

**Your classification must result in one of the following four categories, which you will map to a numeric code:**
*   `0: Vacuum` (The electron beam did not interact with the sample.)
*   `1: Crystalline` (The beam interacted with a region of long-range atomic order.)
*   `2: Amorphous` (The beam interacted with a region of short-range atomic order.)
*   `3: Mixed-State` (The beam interacted with a region containing both crystalline and amorphous phases.)

**REQUIRED OUTPUT FORMAT:**

Your final output **must** be a single, valid JSON object. Do not include any text, explanation, or markdown formatting before or after the JSON block. The JSON object must have exactly two keys: `classification_code` and `description`.

**JSON Structure:**
```json
{
  "classification_code": "integer",
  "description": "string"
}
```

**Instructions for each key:**
1.  **`classification_code`**: This field must contain a single integer from the set `{0, 1, 2, 3}`, corresponding to the classification you determined.
    *   `0` for Vacuum
    *   `1` for Crystalline
    *   `2` for Amorphous
    *   `3` for Mixed-State

2.  **`description`**: In a single, objective paragraph (as a string), provide a detailed analysis of the key visual features in the CBED pattern that justify your classification.
"""

llm_orchestrator = None
consensus_analyzer = None


def initialize_llm():
    global llm_orchestrator, consensus_analyzer
    config = ConfigManager().get_api_keys()
    if config and config.get("api_keys") and not any("your-api-key" in key for key in config["api_keys"]):
        llm_orchestrator = LLMOrchestrator(config["api_keys"], config["base_url"], config["model"])
        consensus_analyzer = ConsensusAnalyzer(llm_orchestrator)


# Initialize FastMCP with default settings
mcp = FastMCP("postgres-mcp")

ResponseType = List[types.TextContent | types.ImageContent | types.EmbeddedResource]


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


@mcp.tool(
    description="Run clustering analysis on a 4D-STEM scan stored in the database. This performs feature extraction, PCA, k-means clustering, and generates montages and XY maps. Also stores cluster assignments in the database for LLM analysis."
)
async def analyze_scan_tool(
    ctx: Context,
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan to analyze."
    ),
    out_root: str = Field(
        description="Output root directory where analysis results will be saved.",
        default=str(data_root() / "results"),
    ),
    k_clusters: int = Field(
        description="Number of clusters for k-means clustering.", default=16
    ),
) -> ResponseType:
    try:
        result = await analyze_scan(await get_sql_driver(), scan_identifier, out_root, k_clusters)
        return format_text_response(json.dumps(result))
    except Exception as error:
        logger.exception("Clustering failed")
        return format_error_response(str(error))


@mcp.tool(
    description="Display classification results for a scan that has already been analyzed. Shows the classification map and cluster montages without re-running analysis."
)
async def display_classification_results(
    ctx: Context,
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan."
    ),
    max_clusters_to_show: int = Field(
        description="Maximum number of cluster montages to display (default: 4)",
        default=4,
    ),
) -> ResponseType:
    """
    Displays the classification results for a scan that has already been analyzed.

    This is a quick way to view classification images without re-running the clustering analysis.
    Use this after running analyze_scan_tool if you want to see the results again.
    """
    try:
        # Use the existing show_classification_overview function
        return await show_classification_overview(
            ctx, scan_identifier, max_clusters_to_show
        )
    except Exception as e:
        logger.error(f"Error displaying classification results: {e}")
        return [types.TextContent(type="text", text=f"❌ Error: {str(e)}")]


@mcp.tool(
    description="Verify cluster assignments are stored in the database for a specific scan. Shows cluster distribution and database storage status."
)
async def verify_cluster_storage_tool(
    scan_name: str = Field(
        description="The name of the scan to check cluster assignments for."
    ),
) -> ResponseType:
    """
    Verify that cluster assignments are properly stored in the database.

    This tool:
    1. Checks if diffraction patterns have cluster_label assigned
    2. Shows cluster distribution statistics
    3. Lists clustering runs for the scan
    4. Verifies database integrity for cluster assignments

    Returns detailed information about cluster storage status.
    """
    try:
        sql_driver = await get_sql_driver()

        # Get clustering runs for this scan
        runs_query = """
            SELECT cr.id, cr.k_value, cr.run_timestamp, cr.algorithm_details,
                   COUNT(dp.id) as patterns_with_clusters
            FROM clustering_runs cr
            JOIN scans s ON cr.scan_id = s.id
            LEFT JOIN diffraction_patterns dp ON dp.clustering_run_id = cr.id
            WHERE s.scan_name = %s
            GROUP BY cr.id, cr.k_value, cr.run_timestamp, cr.algorithm_details
            ORDER BY cr.run_timestamp DESC
        """
        runs_result = await sql_driver.execute_query(runs_query, [scan_name])

        if not runs_result:
            return format_text_response(
                {
                    "scan_name": scan_name,
                    "status": "No clustering runs found",
                    "clustering_runs": [],
                    "cluster_distribution": {},
                }
            )

        # Get cluster distribution for the latest run
        latest_run_id = runs_result[0].cells["id"]

        distribution_query = """
            SELECT dp.cluster_label, COUNT(*) as pattern_count,
                   MIN(rmf.row_index) as min_row, MAX(rmf.row_index) as max_row,
                   MIN(dp.col_index) as min_col, MAX(dp.col_index) as max_col
            FROM diffraction_patterns dp
            JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
            JOIN scans s ON rmf.scan_id = s.id
            WHERE s.scan_name = %s AND dp.clustering_run_id = %s AND dp.cluster_label IS NOT NULL
            GROUP BY dp.cluster_label
            ORDER BY dp.cluster_label
        """
        distribution_result = await sql_driver.execute_query(
            distribution_query, [scan_name, latest_run_id]
        )

        # Get total patterns vs patterns with clusters
        total_patterns_query = """
            SELECT 
                COUNT(*) as total_patterns,
                COUNT(dp.cluster_label) as patterns_with_clusters,
                COUNT(DISTINCT dp.cluster_label) as unique_clusters
            FROM diffraction_patterns dp
            JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
            JOIN scans s ON rmf.scan_id = s.id
            WHERE s.scan_name = %s AND dp.clustering_run_id = %s
        """
        total_result = await sql_driver.execute_query(
            total_patterns_query, [scan_name, latest_run_id]
        )

        # Format results
        clustering_runs = [
            {
                "run_id": row.cells["id"],
                "k_value": row.cells["k_value"],
                "timestamp": str(row.cells["run_timestamp"]),
                "algorithm_details": row.cells["algorithm_details"],
                "patterns_with_clusters": row.cells["patterns_with_clusters"],
            }
            for row in runs_result
        ]

        cluster_distribution = {}
        for row in distribution_result:
            cluster_id = row.cells["cluster_label"]
            cluster_distribution[cluster_id] = {
                "pattern_count": row.cells["pattern_count"],
                "spatial_bounds": {
                    "row_range": [row.cells["min_row"], row.cells["max_row"]],
                    "col_range": [row.cells["min_col"], row.cells["max_col"]],
                },
            }

        stats = total_result[0].cells if total_result else {}

        response_data = {
            "scan_name": scan_name,
            "latest_clustering_run_id": latest_run_id,
            "storage_statistics": {
                "total_patterns": stats.get("total_patterns", 0),
                "patterns_with_clusters": stats.get("patterns_with_clusters", 0),
                "unique_clusters": stats.get("unique_clusters", 0),
                "storage_complete": (
                    stats.get("total_patterns", 0)
                    == stats.get("patterns_with_clusters", 0)
                    if stats.get("total_patterns", 0) > 0
                    else False
                ),
            },
            "clustering_runs": clustering_runs,
            "cluster_distribution": cluster_distribution,
        }

        return format_text_response(response_data)

    except Exception as e:
        logger.error(
            f"Error verifying cluster storage for scan '{scan_name}': {e}",
            exc_info=True,
        )
        return format_error_response(str(e))


async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    initialize_llm()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a custom database config JSON file.",
        default="config/database.json",
    )
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
        await initialize_schema(db_connection.pool)
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
    source_mib_path: str = Field(description="Absolute path to the source .mib file"),
    scan_rows: int | None = Field(default=None, description="Scan rows, if not detected from metadata"),
    scan_columns: int | None = Field(default=None, description="Scan columns, if not detected from metadata"),
    scan_name: str | None = Field(default=None, description="Optional scan name"),
) -> ResponseType:
    """
    When you run this function, you are running the first step of the 4D-STEM data ingestion pipeline. You are a helpful assistant that helps users process and ingest new 4D-STEM scans from .mib files into a PostgreSQL database.
    This is a one-stop tool to process and ingest a new 4D-STEM scan from a .mib file.
    This tool performs a multi-step workflow:
    1. Standardizes file locations by copying the source .mib to a managed 'Raw' directory.
    2. Unpacks the .mib file into a structured folder of .mat files within a 'Data' directory.
    3. Catalogs the entire file structure (scan, .mat files, diffraction points) into the PostgreSQL database.
    """
    try:
        size = (scan_rows, scan_columns) if scan_rows is not None and scan_columns is not None else None
        if (scan_rows is None) != (scan_columns is None):
            raise ValueError("Specify both scan_rows and scan_columns")
        result = await ingest_scan(_unrestricted_driver, source_mib_path, size, scan_name)
        return format_text_response(json.dumps(result))
    except Exception as error:
        logger.exception("Scan import failed")
        return format_error_response(str(error))


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


def preprocess_image(img, top_percent=0.5, eps=1e-8, crop_size=224):
    """
    从中心裁剪到 crop_size × crop_size，
    然后去掉最亮 top_percent 百分比的像素，
    再归一到 [0,1]
    """
    img = img.astype(np.float32)

    # 中心裁剪
    H, W = img.shape
    ch = crop_size // 2
    center_row = H // 2
    center_col = W // 2
    img_cropped = img[
        max(center_row - ch, 0) : min(center_row + ch, H),
        max(center_col - ch, 0) : min(center_col + ch, W),
    ]

    # 如果裁出来不足 crop_size，再补零（padding）
    if img_cropped.shape[0] != crop_size or img_cropped.shape[1] != crop_size:
        padded = np.zeros((crop_size, crop_size), dtype=img.dtype)
        h, w = img_cropped.shape
        padded[:h, :w] = img_cropped
        img_cropped = padded

    # 再做 top_percent 剪切
    if top_percent > 0:
        threshold = np.percentile(img_cropped, 100 - top_percent)
        img_cropped = np.clip(img_cropped, None, threshold)

    mn, mx = img_cropped.min(), img_cropped.max()
    img_cropped = (img_cropped - mn) / (mx - mn + eps)

    return img_cropped


def extract_image_from_mat_file(
    mat_file_path: str, image_in_mat: int, group_number: int, output_path: str
):
    mat_data = scipy.io.loadmat(mat_file_path)
    data = mat_data["data"]
    image = data[image_in_mat - 1]
    image = preprocess_image(image)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, str(group_number)), exist_ok=True)
    output_filename = os.path.join(
        output_path, str(group_number), f"image_{image_in_mat}.png"
    )
    img_uint8 = (image * 255).astype(np.uint8)
    pil_img = pil_Image.fromarray(img_uint8)
    pil_img.save(output_filename)
    return str(output_filename)


@mcp.tool(
    description="Show a specified raw image from the database based on its ID and its group id. Also displays any LLM analysis tags associated with the image."
)
async def show_raw_image(
    image_in_mat: int, mat_number: int, scan_id: int, ctx: Context
) -> ResponseType:
    """
    Retrieves a single image and display it to user based on the provided scan ID and matrix number and group id.
    Also retrieves and displays any LLM analysis tags associated with the image.

    For example, if the user wants to see image 5 from the 10.mat file of scan ID 2, you should provide:
    image_in_mat=5, mat_number=10, scan_id=2.
    """
    logger.info(
        f"Starting show_raw_image with image_in_mat={image_in_mat}, mat_number={mat_number}, scan_id={scan_id}"
    )

    try:
        sql_driver = await get_sql_driver()
        logger.info("Obtained SQL driver")

        # Use parameterized query to get the file path
        rows = await SafeSqlDriver.execute_param_query(
            sql_driver,
            "SELECT id, file_path FROM raw_mat_files WHERE scan_id = {} AND row_index = {};",
            [scan_id, mat_number],
        )

        if not rows:
            logger.warning(
                f"No mat file found for scan_id={scan_id} and row_index={mat_number}"
            )
            raise Exception(
                f"No mat file found for scan_id={scan_id} and row_index={mat_number}"
            )

        mat_file_id = rows[0].cells["id"]
        mat_file_path = rows[0].cells["file_path"]
        logger.info(f"Found mat file: id={mat_file_id}, path={mat_file_path}")

        if not os.path.exists(mat_file_path):
            logger.error(f"Mat file does not exist: {mat_file_path}")
            raise Exception(f"Mat file does not exist: {mat_file_path}")

        image_dir = os.path.join(
            str(os.path.dirname(mat_file_path)), "..", "extracted_images"
        )
        logger.info(f"Extracting image to directory: {image_dir}")
        image_path = extract_image_from_mat_file(
            mat_file_path, image_in_mat, mat_number, image_dir
        )
        logger.info(f"Extracted image to path: {image_path}")
        image_path = str.replace(image_path, "\\\\", "/")
        image_data = (await fetch_images([image_path], ctx))[0]
        logger.info("Fetched image data successfully")

        # Try to get LLM analysis tags for this image
        logger.info("Querying for LLM analysis tags")
        tags_query = """
            SELECT lat.tag_category, lat.tag_value, lat.confidence_score
            FROM llm_analysis_tags lat
            JOIN llm_analysis_results lar ON lat.result_id = lar.id
            JOIN diffraction_patterns dp ON lar.pattern_id = dp.id
            WHERE dp.source_mat_id = %s AND dp.col_index = %s
            ORDER BY lat.created_at DESC
        """

        tags_rows = await sql_driver.execute_query(
            tags_query, [mat_file_id, image_in_mat]
        )
        logger.info(f"Found {len(tags_rows) if tags_rows else 0} tags for this image")

        # Prepare response with both image and tags
        response_items = [
            types.ImageContent(
                type="image", data=image_data.data, mimeType=image_data.mimeType
            )
        ]

        if tags_rows:
            tags_info = "\nLLM Analysis Tags:\n"
            tags_info += "-" * 20 + "\n"
            for row in tags_rows:
                tags_info += f"• {row.cells['tag_category']}: {row.cells['tag_value']} (confidence: {row.cells['confidence_score']:.2f})\n"
            response_items.append(types.TextContent(type="text", text=tags_info))
            logger.info("Added tags to response")
        else:
            response_items.append(
                types.TextContent(
                    type="text", text="\nNo LLM analysis tags found for this image."
                )
            )
            logger.info("No tags found for this image")

        logger.info("Returning image and tags response")
        return response_items

    except Exception as e:
        logger.error(f"Error showing raw image: {e}", exc_info=True)
        raise e


@mcp.tool(
    description="Generate consensus descriptions for all clusters in a scan using LLM analysis. This analyzes 16 representative patterns from each cluster and generates a consensus description for each k category."
)
async def generate_cluster_consensus_tool(
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan to analyze."
    ),
    model: str = Field(
        description="LLM model to use for analysis (optional)", default=None
    ),
) -> ResponseType:
    """
    Generate consensus descriptions for all clusters in a scan using LLM analysis.

    This tool analyzes 16 representative patterns from each cluster and generates
    a consensus description for each k category using Gemini 2.5 Pro.
    Results are stored in the database for future reference.
    """
    logger.info(
        f"Starting generate_cluster_consensus_tool for scan_identifier={scan_identifier}"
    )

    if not consensus_analyzer or not llm_orchestrator:
        logger.error(
            "LLM analysis features are not available. API configuration not found."
        )
        return format_error_response(
            "LLM analysis features are not available. API configuration not found."
        )

    try:
        sql_driver = await get_sql_driver()
        logger.info("Obtained SQL driver")

        # Resolve scan ID
        if isinstance(scan_identifier, int):
            condition_column = "id"
            param = scan_identifier
        else:
            condition_column = "scan_name"
            param = str(scan_identifier)

        logger.info(
            f"Resolving scan with condition_column={condition_column}, param={param}"
        )

        # Get scan information
        scan_query = f"SELECT id, scan_name FROM scans WHERE {condition_column} = {{}}"
        scan_rows = await SafeSqlDriver.execute_param_query(
            sql_driver, scan_query, [param]
        )

        if not scan_rows:
            logger.warning(f"No scan found with identifier: {scan_identifier}")
            return format_error_response(
                f"No scan found with identifier: {scan_identifier}"
            )

        scan_id = scan_rows[0].cells["id"]
        scan_name = scan_rows[0].cells["scan_name"]
        logger.info(f"Found scan: id={scan_id}, name={scan_name}")

        # Get the latest clustering run for this scan
        clustering_query = """
            SELECT id, k_value, run_timestamp
            FROM clustering_runs 
            WHERE scan_id = %s 
            ORDER BY run_timestamp DESC 
            LIMIT 1
        """
        logger.info("Querying for latest clustering run")
        clustering_rows = await sql_driver.execute_query(clustering_query, [scan_id])

        if not clustering_rows:
            logger.warning(f"No clustering run found for scan: {scan_name}")
            return format_error_response(
                f"No clustering run found for scan: {scan_name}"
            )

        clustering_run_id = clustering_rows[0].cells["id"]
        k_value = clustering_rows[0].cells["k_value"]
        logger.info(f"Found clustering run: id={clustering_run_id}, k_value={k_value}")

        logger.info(
            f"Generating consensus for scan '{scan_name}' with {k_value} clusters"
        )

        # Generate consensus for each cluster in parallel (2 at a time)
        logger.info(
            f"Generating consensus descriptions for scan '{scan_name}' with {k_value} clusters using concurrency of 2"
        )
        # First retrieve valid cluster IDs
        cluster_list = []
        for cluster_index in range(k_value):
            # Retrieve cluster database ID
            cluster_query = """
                SELECT id FROM identified_clusters 
                WHERE run_id = %s AND cluster_index = %s
            """
            logger.info(f"Querying cluster_id for cluster_index={cluster_index}")
            rows = await sql_driver.execute_query(
                cluster_query, [clustering_run_id, cluster_index]
            )
            if not rows:
                logger.warning(f"No cluster found for index {cluster_index}")
                continue
            cluster_id = rows[0].cells["id"]
            cluster_list.append((cluster_index, cluster_id))
        consensus_results = []
        # Process all clusters in parallel so all batch images are generated at once
        concurrency = 2
        for i in range(0, len(cluster_list), concurrency):
            batch = cluster_list[i : i + concurrency]
            tasks = []
            for cluster_index, cluster_id in batch:
                logger.info(f"Scheduling analysis for cluster {cluster_index}")
                tasks.append(
                    consensus_analyzer.analyze_cluster_patterns(
                        sql_driver,
                        scan_id,
                        clustering_run_id,
                        cluster_id,
                        cluster_index,
                        model,
                    )
                )
            # Run batch in parallel
            batch_results = await asyncio.gather(*tasks)
            for result in batch_results:
                logger.info(
                    f"Completed consensus for cluster {result['cluster_index']}"
                )
                consensus_results.append(result)
        logger.info(f"Completed analysis for all {len(consensus_results)} clusters")

        # Store results in database
        logger.info("Storing consensus results in database")
        await consensus_analyzer.store_consensus_results(
            sql_driver, scan_id, clustering_run_id, consensus_results
        )
        logger.info("Completed storing consensus results")

        # Format response
        response_text = f"✅ Consensus analysis completed for scan '{scan_name}' ({k_value} clusters)\n\n"
        response_text += "Results:\n"
        for result in consensus_results:
            response_text += f"  Cluster {result['cluster_index']}: {result['pattern_count']} patterns analyzed\n"

        response_text += "\n💡 Use 'show_cluster_consensus' to view detailed results."

        logger.info("Returning success response")
        return format_text_response(response_text)

    except Exception as e:
        logger.error(f"Error generating cluster consensus: {e}", exc_info=True)
        return format_error_response(str(e))


@mcp.tool(
    description="Generate LLM tags for individual diffraction patterns and store them in the database."
)
async def generate_pattern_tags_tool(
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan to analyze."
    ),
    max_patterns: int = Field(
        description="Maximum number of patterns to analyze (default: 100)", default=100
    ),
) -> ResponseType:
    try:
        driver = await get_sql_driver()
        column = "id" if isinstance(scan_identifier, int) else "scan_name"
        scans = await driver.execute_query(f"SELECT scan_name FROM scans WHERE {column} = %s", [scan_identifier])
        if not scans:
            raise ValueError("Scan not found")
        config = ConfigManager().get_api_keys()
        if not config or not config.get("api_keys"):
            raise ValueError("Configure LLM API keys first")
        if max_patterns < 1:
            raise ValueError("max_patterns must be positive")
        pipeline = AnalysisPipeline(driver, config["api_keys"], config["base_url"], config["model"])
        result = await pipeline.analyze_scan_clusters(scans[0].cells["scan_name"], max_patterns_per_cluster=max_patterns, max_total_patterns=max_patterns)
        return format_text_response(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        return format_error_response(str(error))


@mcp.tool(
    description="Show the consensus description for a specific cluster in a scan."
)
async def show_cluster_consensus(
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan."
    ),
    cluster_index: int = Field(
        description="The cluster index to show consensus for (0-based)."
    ),
) -> ResponseType:
    """
    Show the consensus description for a specific cluster in a scan.
    """
    logger.info(
        f"Starting show_cluster_consensus for scan_identifier={scan_identifier}, cluster_index={cluster_index}"
    )

    try:
        sql_driver = await get_sql_driver()

        # Resolve scan ID
        if isinstance(scan_identifier, int):
            condition_column = "s.id"
            param = scan_identifier
        else:
            condition_column = "s.scan_name"
            param = str(scan_identifier)

        # Get consensus analysis results
        query = f"""
            SELECT 
                la.llm_assigned_class,
                la.llm_detailed_features,
                ic.cluster_index,
                s.scan_name
            FROM llm_analyses la
            JOIN identified_clusters ic ON la.cluster_id = ic.id
            JOIN clustering_runs cr ON ic.run_id = cr.id
            JOIN scans s ON cr.scan_id = s.id
            WHERE {condition_column} = {{}} AND ic.cluster_index = {{}}
            ORDER BY cr.run_timestamp DESC
            LIMIT 1
        """

        logger.info("Executing query to get consensus analysis results")
        rows = await SafeSqlDriver.execute_param_query(
            sql_driver, query, [param, cluster_index]
        )

        if not rows:
            logger.warning(
                f"No consensus analysis found for cluster {cluster_index} in scan {scan_identifier}"
            )
            return format_error_response(
                f"No consensus analysis found for cluster {cluster_index} in scan {scan_identifier}"
            )

        result = rows[0].cells
        detailed_features = result["llm_detailed_features"]
        logger.info(f"Found consensus analysis results for cluster {cluster_index}")
        logger.debug(f"Detailed features: {detailed_features}")

        response_text = f"Consensus Analysis for Scan '{result['scan_name']}', Cluster {result['cluster_index']}\\n"
        response_text += "=" * 60 + "\\n"
        response_text += f"Assigned Class: {result['llm_assigned_class']}\\n\\n"
        response_text = f"Consensus Analysis for Scan '{result['scan_name']}', Cluster {result['cluster_index']}\n"
        response_text += "=" * 60 + "\n"
        response_text += f"Assigned Class: {result['llm_assigned_class']}\n\n"
        response_text += (
            f"Description: {detailed_features.get('consensus_description', 'N/A')}\n\n"
        )
        response_text += (
            f"Pattern Count: {detailed_features.get('pattern_count', 'N/A')}\n"
        )
        response_text += f"Dominant Classification Code: {detailed_features.get('dominant_classification_code', 'N/A')}"

        logger.info("Returning consensus analysis results")
        return format_text_response(response_text)

    except Exception as e:
        logger.error(f"Error showing cluster consensus: {e}", exc_info=True)
        return format_error_response(str(e))


@mcp.tool(
    description="Display an image from a local file URL (like file:///path/to/image.png) using the MCP image system."
)
async def show_classification_map(
    ctx: Context,
    file_url: str = Field(
        description="Local file URL starting with 'file://' (e.g., 'file:///tmp/scan_analysis/1/1_xy_map_FINAL.png')"
    ),
) -> Image:
    """
    Displays an image from a local file URL using the MCP image system.

    This function accepts file URLs in the format:
    - file:///absolute/path/to/image.png
    - file://localhost/absolute/path/to/image.png

    The function will extract the local file path and load the image using the MCP image processing system.
    """
    try:
        from urllib.parse import urlparse

        # Parse the file URL
        parsed_url = urlparse(file_url)

        if parsed_url.scheme != "file":
            raise Exception(
                f"Invalid URL scheme. Expected 'file://', got '{parsed_url.scheme}://'"
            )

        # Extract the local file path
        local_file_path = parsed_url.path

        # Handle Windows paths if needed
        if (
            os.name == "nt"
            and local_file_path.startswith("/")
            and ":" in local_file_path[1:3]
        ):
            local_file_path = local_file_path[
                1:
            ]  # Remove leading slash for Windows paths like /C:/...

        logger.info(
            f"Extracting local file path from URL: {file_url} -> {local_file_path}"
        )

        if not os.path.exists(local_file_path):
            raise Exception(f"File does not exist: {local_file_path}")

        if not os.path.isfile(local_file_path):
            raise Exception(f"Path is not a file: {local_file_path}")

        # Check if it's an image file based on extension
        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        }
        file_ext = os.path.splitext(local_file_path)[1].lower()
        if file_ext not in image_extensions:
            raise Exception(
                f"File does not appear to be an image (extension: {file_ext})"
            )

        # Use the MCP image system to load and process the image
        logger.info(f"Loading image using MCP image system: {local_file_path}")
        image_data = (await fetch_images([local_file_path], ctx))[0]

        if image_data is None:
            raise Exception(f"Failed to load image from: {local_file_path}")

        return image_data

    except Exception as e:
        logger.error(f"Error displaying image from local file URL '{file_url}': {e}")
        raise e


@mcp.tool(
    description="Show cluster montage images generated after clustering analysis. Shows representative patterns for each cluster."
)
async def show_cluster_montages(
    ctx: Context,
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan."
    ),
    cluster_id: int = Field(
        description="The cluster ID to show montage for (0-based index)."
    ),
) -> Image:
    """
    Displays montage images for a specific cluster showing representative diffraction patterns.

    Montages are generated during clustering analysis and show sample patterns from each cluster
    to help understand what each cluster represents.
    """
    try:
        sql_driver = await get_sql_driver()

        # Get the clustering run information and results path
        if isinstance(scan_identifier, int):
            condition_column = "s.id"
            param = scan_identifier
        else:
            condition_column = "s.scan_name"
            param = str(scan_identifier)

        query = f"""
            SELECT cr.results_path, s.scan_name
            FROM clustering_runs cr
            JOIN scans s ON cr.scan_id = s.id
            WHERE {condition_column} = {{}}
            ORDER BY cr.run_timestamp DESC
            LIMIT 1
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [param])

        if not rows:
            raise Exception(f"No clustering results found for scan: {scan_identifier}")

        results_path = rows[0].cells["results_path"]
        scan_name = rows[0].cells["scan_name"]

        if not results_path:
            raise Exception(f"No clustering results path found for scan '{scan_name}'")

        # Construct montage file path
        montage_dir = os.path.join(results_path, "montages")
        montage_file = os.path.join(montage_dir, f"cluster_{cluster_id}.png")

        if not os.path.exists(montage_file):
            raise Exception(f"Montage file does not exist: {montage_file}")

        # Normalize path separators
        montage_file = str.replace(montage_file, "\\", "/")
        image_data = (await fetch_images([montage_file], ctx))[0]
        return image_data

    except Exception as e:
        logger.error(f"Error showing cluster montage: {e}")
        raise e


@mcp.tool(
    description="List all available cluster images and classification maps for a scan."
)
async def list_classification_images(
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan."
    ),
) -> ResponseType:
    """
    Lists all available classification-related images for a scan including:
    - Classification map (XY plot)
    - Individual cluster montages
    - File paths and existence status
    """
    try:
        sql_driver = await get_sql_driver()

        # Get scan and clustering information
        if isinstance(scan_identifier, int):
            condition_column = "s.id"
            param = scan_identifier
        else:
            condition_column = "s.scan_name"
            param = str(scan_identifier)

        query = f"""
            SELECT s.id, s.scan_name, s.classification_map_path,
                   cr.id as clustering_run_id, cr.results_path, cr.k_value,
                   cr.run_timestamp
            FROM scans s
            LEFT JOIN clustering_runs cr ON s.id = cr.scan_id
            WHERE {condition_column} = {{}}
            ORDER BY cr.run_timestamp DESC
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [param])

        if not rows:
            return format_text_response(
                f"No scan found with identifier: {scan_identifier}"
            )

        scan_info = rows[0].cells
        scan_name = scan_info["scan_name"]
        classification_map_path = scan_info["classification_map_path"]

        # Prepare response data
        response_data = {
            "scan_id": scan_info["id"],
            "scan_name": scan_name,
            "classification_images": {},
        }

        # Check classification map
        if classification_map_path:
            response_data["classification_images"]["classification_map"] = {
                "path": classification_map_path,
                "exists": os.path.exists(classification_map_path),
                "description": "Overall cluster classification XY map",
            }
        else:
            response_data["classification_images"]["classification_map"] = {
                "path": None,
                "exists": False,
                "description": "Classification map not generated yet",
            }

        # Check cluster montages
        clustering_runs = []
        for row in rows:
            if row.cells["clustering_run_id"]:
                run_data = {
                    "clustering_run_id": row.cells["clustering_run_id"],
                    "k_value": row.cells["k_value"],
                    "timestamp": str(row.cells["run_timestamp"]),
                    "results_path": row.cells["results_path"],
                    "montages": {},
                }

                results_path = row.cells["results_path"]
                k_value = row.cells["k_value"]

                if results_path and k_value:
                    montage_dir = os.path.join(results_path, "montages")

                    # Check each cluster montage
                    for cluster_idx in range(k_value):
                        montage_file = os.path.join(
                            montage_dir, f"cluster_{cluster_idx}.png"
                        )
                        run_data["montages"][f"cluster_{cluster_idx}"] = {
                            "path": montage_file,
                            "exists": os.path.exists(montage_file),
                            "description": f"Montage for cluster {cluster_idx}",
                        }

                clustering_runs.append(run_data)

        response_data["clustering_runs"] = clustering_runs

        return format_text_response(response_data)

    except Exception as e:
        logger.error(f"Error listing classification images: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Show a comprehensive classification overview including the classification map and sample cluster montages for a scan."
)
async def show_classification_overview(
    ctx: Context,
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (integer) or name (string) of the scan."
    ),
    max_clusters_to_show: int = Field(
        description="Maximum number of cluster montages to display (default: 4)",
        default=4,
    ),
) -> ResponseType:
    """
    Displays a comprehensive overview of classification results including:
    - The overall classification map (XY plot)
    - Sample cluster montages for the first few clusters
    - Summary statistics

    This is a convenient tool to quickly see the results of clustering analysis.
    """
    try:
        # First get the classification map
        try:
            classification_map = await show_classification_map(ctx, scan_identifier)
            # Convert Image to ImageContent
            results = [
                types.ImageContent(
                    type="image",
                    data=classification_map.data,
                    mimeType=classification_map.mimeType,
                )
            ]
        except Exception as e:
            results = []
            logger.warning(f"Could not load classification map: {e}")

        # Get scan information and clustering details
        sql_driver = await get_sql_driver()

        if isinstance(scan_identifier, int):
            condition_column = "s.id"
            param = scan_identifier
        else:
            condition_column = "s.scan_name"
            param = str(scan_identifier)

        query = f"""
            SELECT s.scan_name, cr.k_value, cr.run_timestamp,
                   COUNT(dp.id) as total_patterns,
                   COUNT(DISTINCT dp.cluster_label) as assigned_clusters
            FROM scans s
            LEFT JOIN clustering_runs cr ON s.id = cr.scan_id
            LEFT JOIN diffraction_patterns dp ON dp.clustering_run_id = cr.id
            WHERE {condition_column} = {{}}
            GROUP BY s.scan_name, cr.k_value, cr.run_timestamp
            ORDER BY cr.run_timestamp DESC
            LIMIT 1
        """

        rows = await SafeSqlDriver.execute_param_query(sql_driver, query, [param])

        if rows:
            scan_info = rows[0].cells
            scan_name = scan_info["scan_name"]
            k_value = scan_info["k_value"] or 0
            total_patterns = scan_info["total_patterns"] or 0
            assigned_clusters = scan_info["assigned_clusters"] or 0

            # Add summary information as text
            summary_text = f"""
Classification Overview for Scan: {scan_name}
{"=" * 50}
Total K-value (clusters): {k_value}
Total patterns analyzed: {total_patterns}
Successfully assigned clusters: {assigned_clusters}
Analysis timestamp: {scan_info.get("run_timestamp", "N/A")}

Images shown below:
1. Classification Map - Overall spatial distribution of clusters
"""

            # Add cluster montages (limited by max_clusters_to_show)
            clusters_shown = 0
            if k_value > 0:
                for cluster_id in range(min(k_value, max_clusters_to_show)):
                    try:
                        montage = await show_cluster_montages(
                            ctx, scan_identifier, cluster_id
                        )
                        # Convert Image to ImageContent
                        results.append(
                            types.ImageContent(
                                type="image",
                                data=montage.data,
                                mimeType=montage.mimeType,
                            )
                        )
                        clusters_shown += 1
                        summary_text += f"{clusters_shown + 1}. Cluster {cluster_id} Montage - Representative patterns\n"
                    except Exception as e:
                        logger.warning(
                            f"Could not load montage for cluster {cluster_id}: {e}"
                        )

            if k_value > max_clusters_to_show:
                summary_text += f"\n... and {k_value - max_clusters_to_show} more clusters (use show_cluster_montages tool to view individually)"

            # Insert summary at the beginning
            results.insert(0, types.TextContent(type="text", text=summary_text))

        else:
            results.append(
                types.TextContent(
                    type="text",
                    text=f"No clustering analysis found for scan: {scan_identifier}",
                )
            )

        return results

    except Exception as e:
        logger.error(f"Error showing classification overview: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool(description="Check the status of background analysis jobs.")
async def check_analysis_status() -> ResponseType:
    """
    Check the status of background analysis jobs.

    Returns information about currently running analysis tasks or indicates
    that no analysis is currently running.
    """
    from workbench.jobs import JobManager
    return format_text_response(json.dumps(JobManager().list(), ensure_ascii=False))


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


# ========================================================================
# LLM Analysis Tools
# ========================================================================


@mcp.tool(
    description="Run LLM-based analysis on clustered 4D-STEM diffraction patterns for a specific scan"
)
async def run_llm_cluster_analysis(
    scan_name: str = Field(description="Name of the scan to analyze"),
    cluster_id: int = Field(
        default=None,
        description="Optional specific cluster ID to analyze (if not provided, analyzes all clusters)",
    ),
    max_patterns_per_cluster: int = Field(
        default=5, description="Maximum number of patterns to analyze per cluster"
    ),
    batch_size: int = Field(
        default=3, description="Number of patterns to process in parallel batches"
    ),
) -> ResponseType:
    """
    Run complete LLM analysis pipeline on clustered diffraction patterns.
    This tool uses a generic LLM API to analyze diffraction patterns and identify phases/structures.
    """
    try:
        # Get SQL driver
        sql_driver = await get_sql_driver()

        # Load API keys from configuration
        config_manager = ConfigManager()
        api_keys_config = config_manager.get_api_keys()

        if not api_keys_config or "api_keys" not in api_keys_config:
            return format_error_response(
                "API keys not found in configuration. "
                "Please ensure api_keys.json contains 'api_keys' list."
            )

        api_keys = api_keys_config["api_keys"]
        if not isinstance(api_keys, list) or not api_keys:
            return format_error_response(
                "API keys must be provided as a non-empty list in api_keys.json"
            )

        # Get base URL and model from config, with defaults
        base_url = api_keys_config.get("base_url", "https://api.example.com")
        model = api_keys_config.get("model", "default-model")

        # Initialize analysis pipeline
        pipeline = AnalysisPipeline(sql_driver, api_keys, base_url, model)

        # Run the analysis
        logger.info(f"Starting LLM analysis for scan: {scan_name}")
        results = await pipeline.analyze_scan_clusters(
            scan_name=scan_name,
            cluster_id=cluster_id,
            max_patterns_per_cluster=max_patterns_per_cluster,
            batch_size=batch_size,
        )

        # Format results for display
        if results["status"] == "error":
            return format_error_response(f"Analysis failed: {results['message']}")

        result_lines = [
            f"LLM Analysis Results for Scan: {scan_name}",
            "=" * 60,
            f"Status: {results['status']}",
            f"Duration: {results.get('duration_seconds', 0):.1f} seconds",
            f"Clusters Analyzed: {results['clusters_analyzed']}",
            f"Total Patterns Analyzed: {results['total_patterns_analyzed']}",
            "",
        ]

        if results.get("errors"):
            result_lines.extend(
                [
                    "Errors Encountered:",
                    "-" * 20,
                ]
            )
            for error in results["errors"]:
                result_lines.append(f"  • {error}")
            result_lines.append("")

        # Show sample results from each cluster
        if results.get("cluster_results"):
            result_lines.extend(
                [
                    "Cluster Analysis Summary:",
                    "-" * 30,
                ]
            )

            for cluster_id, cluster_result in results["cluster_results"].items():
                if cluster_result["status"] == "success":
                    sample_analysis = cluster_result.get("sample_analysis", {})
                    result_lines.extend(
                        [
                            f"Cluster {cluster_id}:",
                            f"  Patterns: {cluster_result['patterns_analyzed']}",
                            f"  Success Rate: {cluster_result['successful_analyses']}/{cluster_result['patterns_analyzed']}",
                            f"  Phase Type: {sample_analysis.get('phase_type', 'Unknown')}",
                            f"  Quality: {sample_analysis.get('quality', 'Unknown')}",
                            "",
                        ]
                    )
                else:
                    result_lines.extend(
                        [
                            f"Cluster {cluster_id}: FAILED",
                            f"  Error: {cluster_result.get('message', 'Unknown error')}",
                            "",
                        ]
                    )

        result_lines.extend(
            [
                "",
                "Note: Detailed analysis results have been saved to the database.",
                "Use 'get_llm_analysis_summary' to view comprehensive results.",
            ]
        )

        return format_text_response("\n".join(result_lines))

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return format_error_response(f"LLM analysis failed: {str(e)}")


@mcp.tool(description="Get a summary of all LLM analyses performed on a scan")
async def get_llm_analysis_summary(
    scan_name: str = Field(description="Name of the scan to get analysis summary for"),
) -> ResponseType:
    """
    Retrieve a comprehensive summary of all LLM analyses performed on a scan's clusters.
    """
    try:
        # Get SQL driver
        sql_driver = await get_sql_driver()

        # Load API keys (just for pipeline initialization, won't be used for this read operation)
        config_manager = ConfigManager()
        api_keys_config = config_manager.get_api_keys()

        if api_keys_config:
            api_keys_list = api_keys_config.get("api_keys", ["dummy"])
            base_url = api_keys_config.get("base_url", "https://api.example.com")
            model = api_keys_config.get("model", "default-model")
        else:
            # Dummy values for read operations that don't need actual API calls
            api_keys_list = ["dummy"]
            base_url = "https://api.example.com"
            model = "default-model"

        # Initialize pipeline to use its summary method
        pipeline = AnalysisPipeline(sql_driver, api_keys_list, base_url, model)

        # Get analysis summary
        summary = await pipeline.get_analysis_summary(scan_name)

        if "error" in summary:
            return format_error_response(
                f"Failed to get analysis summary: {summary['error']}"
            )

        if summary["total_clusters_analyzed"] == 0:
            return format_text_response(f"No LLM analyses found for scan: {scan_name}")

        # Format summary for display
        result_lines = [
            f"LLM Analysis Summary for Scan: {scan_name}",
            "=" * 60,
            f"Total Clusters Analyzed: {summary['total_clusters_analyzed']}",
            "",
            "Detailed Results:",
            "-" * 20,
        ]

        for analysis in summary["analyses"]:
            result_lines.extend(
                [
                    f"Analysis ID: {analysis['analysis_id']}",
                    f"  Cluster ID: {analysis['cluster_id']}",
                    f"  Representative Patterns: {analysis['representative_patterns_count']}",
                    f"  Total Patterns in Cluster: {analysis['total_patterns_in_cluster']}",
                    f"  LLM Classification: {analysis['llm_assigned_class']}",
                    f"  Analysis Date: {analysis['created_at']}",
                    "",
                ]
            )

        return format_text_response("\n".join(result_lines))

    except Exception as e:
        logger.error(f"Failed to get LLM analysis summary: {e}")
        return format_error_response(f"Failed to get analysis summary: {str(e)}")


@mcp.tool(
    description="List and inspect JSON analysis results from batch image processing"
)
async def list_json_analysis_results(
    results_folder: str = Field(
        default="/tmp/results_llm",
        description="Folder containing JSON analysis results",
    ),
    show_details: bool = Field(
        default=False, description="Show detailed content of each JSON file"
    ),
) -> ResponseType:
    """
    List and optionally show details of JSON analysis results generated by batch image processing.
    """
    try:
        import json
        from pathlib import Path
        from datetime import datetime

        results_path = Path(results_folder)

        if not results_path.exists():
            return format_text_response(f"Results folder not found: {results_folder}")

        # Find JSON files
        json_files = list(results_path.glob("*_analysis.json"))

        if not json_files:
            return format_text_response(
                f"No analysis JSON files found in {results_folder}"
            )

        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        result_lines = [
            f"JSON Analysis Results in {results_folder}",
            "=" * 60,
            f"Total Files: {len(json_files)}",
            "",
        ]

        successful_analyses = 0
        failed_analyses = 0

        for json_file in json_files:
            try:
                # Read JSON content
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                status = data.get("status", "unknown")
                image_name = Path(data.get("image_file", "unknown")).name

                if status == "success":
                    successful_analyses += 1
                    status_icon = "✓"
                else:
                    failed_analyses += 1
                    status_icon = "✗"

                # Basic file info
                file_size = json_file.stat().st_size
                mod_time_timestamp = json_file.stat().st_mtime
                mod_time = datetime.fromtimestamp(mod_time_timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                result_lines.extend(
                    [
                        f"{status_icon} {json_file.name}",
                        f"  Image: {image_name}",
                        f"  Status: {status}",
                        f"  Size: {file_size} bytes",
                        f"  Modified: {mod_time}",
                    ]
                )

                if show_details and status == "success":
                    analysis = data.get("analysis", {})
                    result_lines.extend(
                        [
                            f"  Classification: {analysis.get('image_classification', 'N/A')}",
                            f"  Confidence: {analysis.get('confidence', 'N/A')}",
                            f"  Content: {analysis.get('content_description', 'N/A')[:100]}...",
                        ]
                    )
                elif show_details and status == "error":
                    result_lines.append(
                        f"  Error: {data.get('error', 'Unknown error')}"
                    )

                result_lines.append("")

            except Exception as e:
                result_lines.extend(
                    [
                        f"✗ {json_file.name} (READ ERROR)",
                        f"  Error reading file: {str(e)}",
                        "",
                    ]
                )
                failed_analyses += 1

        # Add summary
        result_lines.extend(
            [
                "Summary:",
                "-" * 20,
                f"Successful Analyses: {successful_analyses}",
                f"Failed Analyses: {failed_analyses}",
                f"Total Files: {len(json_files)}",
            ]
        )

        if not show_details:
            result_lines.extend(
                ["", "Use show_details=True to see analysis content for each file."]
            )

        return format_text_response("\n".join(result_lines))

    except Exception as e:
        logger.error(f"Failed to list JSON results: {e}")
        return format_error_response(f"Failed to list JSON results: {str(e)}")


@mcp.tool(description="Get detailed LLM analysis results for a specific cluster")
async def get_cluster_llm_details(
    scan_name: str = Field(description="Name of the scan"),
    cluster_id: int = Field(
        description="ID of the cluster to get detailed results for"
    ),
) -> ResponseType:
    """
    Get detailed LLM analysis results and individual pattern analyses for a specific cluster.
    """
    try:
        sql_driver = await get_sql_driver()

        # Get detailed analysis for the cluster
        detail_sql = """
            SELECT la.id, la.cluster_id, la.representative_patterns_count,
                   la.llm_assigned_class, la.llm_detailed_features, la.analysis_timestamp,
                   COUNT(dp.id) as total_patterns_in_cluster
            FROM llm_analyses la
            JOIN diffraction_patterns dp ON la.cluster_id = dp.cluster_label
            JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
            JOIN scans s ON rmf.scan_id = s.id
            WHERE s.scan_name = %s AND la.cluster_id = %s
            GROUP BY la.id, la.cluster_id, la.representative_patterns_count,
                     la.llm_assigned_class, la.llm_detailed_features, la.analysis_timestamp;
        """

        result = await sql_driver.execute_query(detail_sql, [scan_name, cluster_id])

        if not result:
            return format_text_response(
                f"No LLM analysis found for cluster {cluster_id} in scan {scan_name}"
            )

        analysis = result[0].cells

        # Parse detailed features if available
        import json

        detailed_features = {}
        try:
            if analysis["llm_detailed_features"]:
                detailed_features = json.loads(analysis["llm_detailed_features"])
        except Exception:
            pass

        # Format detailed results
        result_lines = [
            f"Detailed LLM Analysis for Cluster {cluster_id}",
            "=" * 60,
            f"Scan: {scan_name}",
            f"Analysis ID: {analysis['id']}",
            f"Analysis Date: {analysis['analysis_timestamp']}",
            f"Representative Patterns Analyzed: {analysis['representative_patterns_count']}",
            f"Total Patterns in Cluster: {analysis['total_patterns_in_cluster']}",
            "",
            "LLM Classification Results:",
            "-" * 30,
            f"Phase Type: {analysis['llm_assigned_class']}",
            "",
        ]

        if detailed_features:
            result_lines.extend(
                [
                    "Detailed Analysis Features:",
                    "-" * 30,
                ]
            )
            for key, value in detailed_features.items():
                if key != "phase_type":  # Already shown above
                    result_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            result_lines.append("")

        # Get representative patterns used in analysis
        rep_patterns_sql = """
            SELECT lrp.pattern_id, lrp.selection_reason, rmf.row_index, dp.col_index
            FROM llm_representative_patterns lrp
            JOIN llm_analyses la ON lrp.analysis_id = la.id
            JOIN diffraction_patterns dp ON lrp.pattern_id = dp.id
            JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
            JOIN scans s ON rmf.scan_id = s.id
            WHERE s.scan_name = %s AND la.cluster_id = %s
            ORDER BY rmf.row_index, dp.col_index;
        """

        rep_result = await sql_driver.execute_query(
            rep_patterns_sql, [scan_name, cluster_id]
        )

        if rep_result:
            result_lines.extend(
                [
                    "Representative Patterns Used:",
                    "-" * 30,
                ]
            )
            for rep_pattern in rep_result:
                result_lines.append(
                    f"Pattern ID {rep_pattern.cells['pattern_id']}: "
                    f"Row {rep_pattern.cells['row_index']}, "
                    f"Col {rep_pattern.cells['col_index']} "
                    f"({rep_pattern.cells['selection_reason']})"
                )

        return format_text_response("\n".join(result_lines))

    except Exception as e:
        logger.error(f"Failed to get cluster LLM details: {e}")
        return format_error_response(f"Failed to get cluster details: {str(e)}")


@mcp.tool(
    description="Test LLM connectivity by analyzing a local image file with the configured API"
)
async def test_llm_analysis(
    ctx: Context,
    image_path: str = Field(description="Path to the image file to analyze"),
    model: str = Field(description="Model to use (optional)", default=None),
) -> ResponseType:
    """
    Test LLM analysis on a local image file.
    This is a simple test to verify LLM connectivity and functionality.
    """
    logger.info(f"Starting test_llm_analysis for image_path={image_path}")

    if not llm_orchestrator:
        logger.error("LLM orchestrator not available")
        error_msg = (
            "LLM analysis features are not available. Please check API configuration."
        )
        if (ConfigManager().project_root / "config/api_keys.json").exists():
            error_msg += "\n\nTip: Check that config/api_keys.json contains valid API keys instead of placeholder values."
        return format_error_response(error_msg)

    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return format_error_response(f"Image file not found: {image_path}")

        logger.info(f"File exists, calling LLM orchestrator")

        # Use the JSON prompt template
        result = await llm_orchestrator.analyze_single_image(
            image_path, JSON_PROMPT_TEMPLATE, model
        )

        logger.info(
            f"LLM analysis completed. Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
        )

        # Format the response
        if isinstance(result, dict) and "error" not in result:
            response_text = f"✅ LLM Analysis Test Successful!\n\n"
            response_text += f"Image: {image_path}\n"
            response_text += f"Model: {model or llm_orchestrator.model}\n\n"
            response_text += f"Results:\n"

            if "classification_code" in result:
                class_names = {
                    0: "Vacuum",
                    1: "Crystalline",
                    2: "Amorphous",
                    3: "Mixed-State",
                }
                class_name = class_names.get(result["classification_code"], "Unknown")
                response_text += f"  Classification: {result['classification_code']} ({class_name})\n"

            if "description" in result:
                response_text += f"  Description: {result['description']}\n"

            if "timestamp" in result:
                response_text += f"  Timestamp: {result['timestamp']}\n"

            logger.info("Returning successful test result")
            return format_text_response(response_text)
        else:
            logger.error(f"LLM analysis failed: {result}")
            return format_error_response(f"LLM analysis failed: {result}")

    except Exception as e:
        logger.error(f"Error in test_llm_analysis: {e}", exc_info=True)
        return format_error_response(f"Error in test LLM analysis: {e}")


@mcp.tool(
    description="Regenerate the classification map using LLM-assigned classes (groups by classification_code)."
)
async def regenerate_classification_map_tool(
    ctx: Context,
    scan_identifier: Union[int, str] = Field(
        description="The unique ID (int) or name (str) of the scan."
    ),
    out_root: str = Field(
        description="Output root directory for regenerated map.",
        default=str(data_root() / "results"),
    ),
    clustering_run_id: Optional[int] = Field(
        description="Specific clustering run ID (defaults to latest)", default=None
    ),
) -> Image:
    """
    Regenerate the XY classification map based on latest LLM results.
    """
    sql_driver = await get_sql_driver()
    try:
        # Try to derive a friendly scan name for the output filename
        if isinstance(scan_identifier, int):
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                "SELECT scan_name FROM scans WHERE id = {}",
                [scan_identifier],
            )
            scan_name = rows[0].cells["scan_name"] if rows else str(scan_identifier)
        else:
            scan_name = str(scan_identifier)

        # Ensure output directory exists
        os.makedirs(out_root, exist_ok=True)

        # Build a full output filename for the regenerated map
        image_filename = f"{scan_name}_xy_map_REGENERATED.png"
        full_image_path = os.path.join(out_root, image_filename)

        # Call shared function which expects a full file path
        saved_path = await regenerate_classification_map(
            sql_driver, scan_identifier, full_image_path, clustering_run_id
        )

        # Verify the file was created
        if not os.path.isfile(saved_path):
            raise Exception(
                f"Regenerated classification map not found at: {saved_path}"
            )

        # Load the saved image using the MCP image pipeline so we return a proper Image object
        image_obj = (await fetch_images([saved_path], ctx))[0]
        if image_obj is None:
            raise Exception(
                f"Failed to load regenerated classification map from: {saved_path}"
            )
        return image_obj

    except Exception as e:
        logger.error(f"Error in regenerate_classification_map_tool: {e}", exc_info=True)
        raise
