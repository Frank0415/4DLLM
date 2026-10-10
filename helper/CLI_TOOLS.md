# CLI tools

Run commands from the repository root. Browser, CLI, and MCP use the same import and clustering core.

## Analyze an imported scan

```bash
uv run --locked python helper/analyze_scan_cli.py --scan-id 1
uv run --locked python helper/analyze_scan_cli.py --scan-name sample --k-clusters 8 --seed 0 --device cpu
```

Use `--config` for a custom database configuration and `--out-root` for another result root. Results default to `workspace/results/`, with a new directory for each run. `--job-id` remains accepted for older callers; CLI execution does not create a browser task record.

## List scans

```bash
uv run --locked python helper/list_scans.py
```

## Import external cluster arrays

```bash
uv run --locked python helper/store_clustering_results.py --scan-id 1 --npz-file /path/to/results.npz
```

The NPZ must contain integer `labels`, `xs`, and `ys`, cover the entire scan exactly once, and use one-based row/column coordinates. Imported arrays create a **new** run and result directory; they do not overwrite an earlier run. Optional `--clustering-run-id` is retained as source provenance, not used as a destination to overwrite. Representative montages are not generated for external arrays.

## Interactive NPY workflow

```bash
uv run --locked python pre_class_v2.py /path/to/npy-directory --out-root results --k-clusters 16
```

This legacy interactive workflow shares the clustering implementation in `postgres_mcp/analyze/class_base.py`. It asks for human labels and writes them to the NPY input, retaining its original workflow.

## Database setup

```bash
docker compose -f docker/docker-compose.yml up -d postgres_db
uv run --locked python setup_database.py
```

Initialization creates or upgrades missing schema objects without deleting existing scan data. Database credentials are in `config/database.json`; see [workbench documentation](../docs/documentation.md).
