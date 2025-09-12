import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages database configuration loading from JSON files with a fallback mechanism.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the ConfigManager.
        
        Args:
            config_path: Optional path to a specific config JSON file.
        """
        # 动态确定项目根目录，然后构建配置文件的标准路径
        self.project_root = os.getcwd()
        
        self.primary_config_path = os.path.join(self.project_root,config_path)
        self.fallback_config_path = os.path.join(self.project_root, "config", "db_config.json")
        self.fallfallback_config_path = os.path.join(self.project_root, "config", "db_config_example.json")
        self.config: Dict[str, Any] = {}

    def load_config(self) -> None:
        """
        Loads configuration from the primary path, falling back to the example if not found.
        """
        print(os.path.join(self.project_root,"config", self.primary_config_path))
        if os.path.exists(self.primary_config_path):
            logger.info(f"Loading database configuration from: {self.primary_config_path}")
            with open(self.primary_config_path, 'r') as f:
                self.config = json.load(f)
        elif os.path.exists(os.path.join(self.project_root, self.primary_config_path)):
            logger.warning(f"Primary config not found. Falling back to: {os.path.join(self.project_root, self.primary_config_path)}")
            with open(os.path.join(self.project_root, self.primary_config_path), 'r') as f:
                self.config = json.load(f)
        elif os.path.exists(self.fallback_config_path):
            logger.warning(f"Primary config not found. Falling back to: {self.fallback_config_path}")
            with open(self.fallback_config_path, 'r') as f:
                self.config = json.load(f)
        elif os.path.exists(self.fallfallback_config_path):
            logger.warning(f"Primary and fallback config not found. Falling back to example: {self.fallfallback_config_path}")
            with open(self.fallfallback_config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(
                "No database configuration file found. Please create 'config/db_config.json' or 'config/db_config_example.json'."
            )

    def get_database_url(self) -> Optional[str]:
        """
        Constructs the DATABASE_URI connection string from the loaded config.
        """
        if not self.config:
            self.load_config()
            
        required_keys = ["user", "password", "host", "port", "dbname"]
        print(self.config)
        if not all(key in self.config for key in required_keys):
            logger.error("Configuration file is missing one or more required keys: " + ", ".join(required_keys))
            return None
            
        return (
            f"postgresql://{self.config['user']}:{self.config['password']}@"
            f"{self.config['host']}:{self.config['port']}/{self.config['dbname']}"
        )