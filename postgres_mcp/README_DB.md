# PostgreSQL storage

Create the database first, then initialize or upgrade schema objects with:

```bash
uv run --locked python setup_database.py --config config/database.json
```

`SQL/schema.sql` is non-destructive: tables, columns, indexes, and foreign keys are added if absent. It does not drop tables or erase scans. Schema initialization takes a PostgreSQL advisory lock so application processes do not apply it simultaneously.

The browser can save database configuration in Settings. `DATABASE_URI` overrides the configuration file. The bundled Docker database uses `postgres`, password `1234`, and database `4dllm`, listening on loopback.

Scan metadata includes the managed source MIB, SHA-256, and detected dimensions. Pixel data remain in files; PostgreSQL stores scan and pattern references. Each clustering run has parameters, its own result directory, and historical pattern assignments. LLM results and manual `cluster_annotations` remain separate.

Import and clustering catalog writes use a single transaction and bulk COPY. Individual `SqlDriver.execute_query` calls also use transactions. Existing scan paths continue to work after upgrade.

Database tuning and health-analysis modules inherited from upstream were removed because their MCP entry points were disabled. The SQL driver and read-only SQL validator remain.
