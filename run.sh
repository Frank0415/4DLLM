#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

command -v uv >/dev/null || { echo 'Install uv before starting 4DLLM.'; exit 1; }

if [[ ! -f config/database.json && -z "${DATABASE_URI:-}" && -z "${FOURDLLM_CONFIG:-}" ]]; then
    cp config/db_config_example.json config/database.json
    chmod 600 config/database.json
    if command -v docker >/dev/null; then
        docker compose -f docker/docker-compose.yml up -d postgres_db
    else
        echo 'Configure an existing PostgreSQL instance in Settings, or start docker/docker-compose.yml separately.'
    fi
fi

if [[ ! -f frontend/dist/index.html ]]; then
    command -v npm >/dev/null || { echo 'Install Node.js and npm to build the browser interface.'; exit 1; }
    (cd frontend && npm ci && npm run build)
fi

exec uv run --locked python -m workbench "$@"
