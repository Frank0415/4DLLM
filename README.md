# 4DLLM

[中文](README_cn.md) · English

A local browser workbench for importing, exploring, and analyzing 4D-STEM scans. The Python analysis core is shared by the browser application, CLI, and MCP server.

- Import a MIB file or a directory, retain companion HDR metadata, and preview diffraction patterns.
- Explore scan intensity and cluster maps; click a scan position to inspect its diffraction pattern.
- Run polar feature extraction, PCA, and K-Means on CPU or CUDA.
- Keep each clustering run, its parameters, and its result files separately.
- Request optional Gemini-compatible LLM suggestions and record human annotations independently.
- Follow a persistent task queue with progress, cancellation requests, retry, logs, and downloadable reports.

## Start

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), Node.js/npm for the initial frontend build, and PostgreSQL 17+.

```bash
git clone git@github.com:Frank0415/4DLLM.git
cd 4DLLM
./run.sh
```

The launcher builds the UI when needed and opens `http://127.0.0.1:8765`. On the first launch, if Docker is available, it starts the bundled PostgreSQL service. Otherwise, configure an existing database in **Settings**. The database must already exist; table creation and upgrades preserve existing records.

The bundled database uses `postgres / 1234` and database `4dllm`, matching `config/db_config_example.json`. Configuration files containing credentials are ignored by Git.

```bash
# Start the database independently
docker compose -f docker/docker-compose.yml up -d postgres_db

# Run without opening a browser
./run.sh --no-browser --port 8765

# MCP server
uv run --locked python main.py --config config/database.json
```

## Import and analyze

1. Configure data directories and PostgreSQL in Settings.
2. Select a MIB file in **Import**, inspect its first pattern, and confirm scan rows/columns. Directory import queues the MIB files directly inside that directory.
3. Open **Scans**, select K and the compute device, and submit clustering.
4. Explore the resulting clusters and historical runs. Optionally configure LLM keys, request suggestions, and save human classifications.

Supported MIB storage formats: `U08`, `U16`, and `U32`; detector layouts: `1x1`, `2x2`, and `2x2G`. Packed `R64` is not supported. A scan shape must match the frame count exactly. Keep the `.hdr` beside its `.mib`; missing dimensions can be entered manually.

Cluster numbers are neutral groups. Materials classes are stored as LLM suggestions or human annotations, never inferred from the cluster number.

See [workflow and API documentation](docs/documentation.md), [CLI usage](helper/CLI_TOOLS.md), and [Chinese documentation](docs/documentation_zh.md).

## Storage and optional simulation

New data and results live under `workspace/` by default. Set `FOURDLLM_DATA_DIR` to change the storage directory, `FOURDLLM_CONFIG` to select a configuration file, or `DATABASE_URI` to supply the database connection. Existing imported scan paths remain usable.

CIF management and simulation remain available through MCP. Install simulation dependencies with `uv sync --extra simulation`. CIF pattern comparison is currently unimplemented and returns `not_implemented`.

## License and acknowledgments

MIT; see [LICENSE](LICENSE). This project incorporates work from [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp) and [ia-programming/mcp-images](https://github.com/ia-programming/mcp-images); their license notices remain in [LICENSES](LICENSES).
