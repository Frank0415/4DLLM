#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f config/api_keys.json ]]; then
    cp config/api_keys_example.json config/api_keys.json
    chmod 600 config/api_keys.json
fi
printf '%s\n' 'Configure the Gemini-compatible base URL, model, and API keys in the browser Settings page or config/api_keys.json.'
