# 4DLLM

面向4DSTEM的微观材料核心分类，数据分析与关系构建工具。

---

此项目使用了 [postgres-mcp](https://github.com/crystaldba/postgres-mcp), 原始许可证位于 [LICENSE-postgres-mcp](LICENSES/LICENSE-postgres-mcp)。

## 数据库设置

在运行分析工具之前，需要设置PostgreSQL数据库：

1. 确保PostgreSQL服务器正在运行
2. 复制配置文件：`cp config/db_config_example.json config/database.json`
3. 编辑 `config/database.json` 并填入您的数据库凭据
4. 初始化数据库表：`python postgres_mcp/init_db.py`

更多详细信息，请参阅 [postgres_mcp/README_DB.md](postgres_mcp/README_DB.md)。