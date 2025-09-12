# 4D-STEM Scan Analysis CLI Tools

This directory contains standalone CLI tools for analyzing 4D-STEM scans without the overhead of the MCP/LLM connection. These tools can help identify performance bottlenecks and avoid connection timeouts.

## Tools

### 1. analyze_scan_cli.py

Standalone tool for analyzing 4D-STEM scans directly from the terminal.

**Usage:**
```bash
# Analyze by scan ID
python analyze_scan_cli.py --scan-id 1

# Analyze by scan name
python analyze_scan_cli.py --scan-name "sample_scan"

# With custom parameters
python analyze_scan_cli.py --scan-id 2 --out-root "./results" --k-clusters 8 --device "cpu"
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
python list_scans.py
```

## Prerequisites

1. Database properly configured with `config/database.json`
2. Required database tables created (use `postgres_mcp/init_db.py`)
3. Scan data ingested using the MCP ingestion tools

## Troubleshooting

If you encounter connection issues:
1. Verify database credentials in `config/database.json`
2. Ensure PostgreSQL is running
3. Check that the database tables exist
4. Confirm scan data is properly ingested

## Performance Benefits

These CLI tools bypass the MCP server overhead and connect directly to the database, which should:
- Reduce latency
- Avoid connection timeouts
- Provide better error messages
- Allow for easier debugging of performance bottlenecks