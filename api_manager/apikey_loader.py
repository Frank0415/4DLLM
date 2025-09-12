import json

def load_api_config(config_path: str = "api_keys.json"):
    """加载API配置文件

    Returns:
        Optional[dict[str, Union[list[str], str]]]: 包含 api_keys (list[str]), base_url (str), model (str) 的配置字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到")
        return None
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        return None


if __name__ == "__main__":
    config = load_api_config("config/api_keys.json")
    if config:
        api_keys = config["api_keys"]
        base_url = config["base_url"]
        model = config["model"]
        print(f"Loaded {api_keys} API keys, base_url: {base_url}, model: {model}")
        print(f"{type(api_keys), type(base_url), type(model)}")
