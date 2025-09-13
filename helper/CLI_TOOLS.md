# 4D-STEM Scan Analysis CLI Tools

This directory contains standalone CLI tools for analyzing 4D-STEM scans without the overhead of the MCP/LLM connection. These tools can help identify performance bottlenecks and avoid connection timeouts.

## Tools

### 1. analyze_scan_cli.py

Standalone tool for analyzing 4D-STEM scans directly from the terminal.

**Usage:**
```bash
# Analyze by scan ID
uv run analyze_scan_cli.py --scan-id 1

# Analyze by scan name
uv run analyze_scan_cli.py --scan-name "sample_scan"

# With custom parameters
uv run analyze_scan_cli.py --scan-id 2 --out-root "./results" --k-clusters 8 --device "cpu"
```

**Options:**
- `--scan-id`: Scan ID (integer) to analyze
- `--scan-name`: Scan name (string) to analyze
- `--out-root`: Output directory for results (default: /tmp/scan_analysis)
- `--k-clusters`: Number of clusters for k-means (default: 16)
- `--seed`: Random seed for reproducibility (default: 0)
- `--device`: Torch device ('cpu', 'cuda', etc.) or None to auto-detect
- `--config`: Path to database config file (default: config/database.json)
- `--verbose`: Enable verbose logging

### 2. list_scans.py

Simple tool to test database connection and list available scans.

**Usage:**
```bash
uv run list_scans.py
```

### 3. store_clustering_results.py

Tool for storing k-means clustering results directly in the database for fast querying.

**Usage:**
```bash
# Store results by scan ID
uv run store_clustering_results.py --scan-id 1 --clustering-run-id 100 --npz-file "/path/to/results.npz"

# Store results by scan name
uv run store_clustering_results.py --scan-name "sample_scan" --clustering-run-id 101 --npz-file "./cluster_results.npz"
```

**Options:**
- `--scan-id`: Scan ID (integer) to associate with results
- `--scan-name`: Scan name (string) to associate with results
- `--clustering-run-id`: ID of the clustering run to associate with these results
- `--npz-file`: Path to the .npz file containing clustering results (labels, xs, ys)
- `--config`: Path to database config file (default: config/database.json)
- `--verbose`: Enable verbose logging

## Docker Environment Scripts

For Docker environments, convenience scripts are provided:

### docker_update_schema.sh

Updates the database schema to support cluster result storage:
```bash
./docker_update_schema.sh
```

### docker_test_cluster_storage.sh

Tests the cluster storage functionality:
```bash
./docker_test_cluster_storage.sh
```

### docker_store_clustering_results.sh

Stores clustering results in Docker environment:
```bash
./docker_store_clustering_results.sh <scan_identifier> <clustering_run_id> <npz_file_path> [config_path]
```

## Prerequisites

1. Database properly configured with `config/database.json`
2. Required database tables created (use `postgres_mcp/init_db.py`)
3. Scan data ingested using the MCP ingestion tools
4. Database schema updated for cluster result storage (use `postgres_mcp/update_schema_for_clusters.sql` or `docker_update_schema.sh`)

## Database Schema Updates

The clustering results are stored directly in the `diffraction_patterns` table with two new columns:
- `cluster_label`: Integer representing the cluster assignment (0 to k-1)
- `clustering_run_id`: Reference to the clustering run that generated these labels

This enables fast SQL queries like:
```sql
-- Get all patterns classified as cluster 3
SELECT * FROM diffraction_patterns WHERE cluster_label = 3;

-- Get cluster distribution
SELECT cluster_label, COUNT(*) FROM diffraction_patterns 
WHERE clustering_run_id = 100 
GROUP BY cluster_label;

-- Spatial queries
SELECT dp.cluster_label, rmf.row_index as x, dp.col_index as y
FROM diffraction_patterns dp
JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
WHERE dp.clustering_run_id = 100;
```

## Troubleshooting

If you encounter connection issues:
1. Verify database credentials in `config/database.json`
2. Ensure PostgreSQL is running
3. Check that the database tables exist
4. Confirm scan data is properly ingested
5. Ensure database schema has been updated for cluster storage

## Performance Benefits

These CLI tools bypass the MCP server overhead and connect directly to the database, which should:
- Reduce latency
- Avoid connection timeouts
- Provide better error messages
- Allow for easier debugging of performance bottlenecks
- Enable fast querying of clustering results directly from the database