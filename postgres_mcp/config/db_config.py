import json
import os
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigManager:
    def __init__(self, config_path=None):
        self.project_root = PROJECT_ROOT
        path = Path(config_path or os.environ.get("FOURDLLM_CONFIG", "config/database.json"))
        self.primary_config_path = path if path.is_absolute() else PROJECT_ROOT / path
        self.config = {}

    def load_config(self):
        candidates = [self.primary_config_path]
        if self.primary_config_path == PROJECT_ROOT / "config/database.json":
            candidates.append(PROJECT_ROOT / "config/db_config.json")
        for path in candidates:
            if path.is_file():
                self.config = json.loads(path.read_text(encoding="utf-8"))
                return
        raise FileNotFoundError(f"Database configuration not found: {self.primary_config_path}")

    def get_database_url(self):
        if os.environ.get("DATABASE_URI"):
            return os.environ["DATABASE_URI"]
        if not self.config:
            self.load_config()
        required = ("user", "password", "host", "port", "dbname")
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing database settings: {', '.join(missing)}")
        cfg = self.config
        host = str(cfg["host"])
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return (f"postgresql://{quote(str(cfg['user']), safe='')}:"
                f"{quote(str(cfg['password']), safe='')}@{host}:{int(cfg['port'])}/"
                f"{quote(str(cfg['dbname']), safe='')}")

    def get_api_keys(self):
        for path in (PROJECT_ROOT / "config/api_keys.json", PROJECT_ROOT / "api_manager/api_keys.json"):
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None
