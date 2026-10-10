# 4DLLM

中文 · [English](README.md)

本地浏览器中的 4D-STEM 数据工作台。导入、聚类和 LLM 分析由同一套 Python 核心实现，浏览器、CLI 和 MCP 共用。

- 选择单个 MIB 文件或目录，保留配套 HDR，完成转换与入库。
- 查看扫描总强度图和聚类空间图，点击坐标查看原始衍射图案。
- 在 CPU 或 CUDA 上运行极坐标特征提取、PCA 和 K-Means。
- 每次聚类保留独立的参数、结果目录和历史记录。
- 可选的 Gemini 兼容 LLM 建议，与人工确认分别保存。
- 查看任务进度、日志和报告，取消或重新提交任务。

## 启动

需要 Python 3.13+、[uv](https://docs.astral.sh/uv/)、PostgreSQL 17+。首次构建页面需要 Node.js 和 npm。

```bash
git clone git@github.com:Frank0415/4DLLM.git
cd 4DLLM
./run.sh
```

启动脚本首次运行时构建页面，并打开 `http://127.0.0.1:8765`。如果首次启动时已有 Docker，会启动仓库提供的 PostgreSQL；否则在页面的“设置”中连接已有数据库。数据库本身须事先创建，工作台负责建表和升级，保留已有扫描与结果。

仓库内数据库默认账号为 `postgres`、密码为 `1234`、数据库为 `4dllm`，与数据库配置示例一致。真实数据库配置和 API 密钥不进入 Git。

```bash
# 单独启动数据库
docker compose -f docker/docker-compose.yml up -d postgres_db

# 自定义端口，不自动打开浏览器
./run.sh --no-browser --port 8765

# 原有 MCP 入口
uv run --locked python main.py --config config/database.json
```

## 使用流程

1. 在“设置”中填写 PostgreSQL，并指定允许浏览的数据目录。
2. 在“导入扫描”中选择 MIB，查看预览和元数据，确认行列数后导入。“导入整个目录”处理当前目录中的 MIB 文件，不递归子目录。
3. 在“扫描与分析”中选择 K、随机种子和计算设备，运行聚类。
4. 选择历史运行，查看各簇代表图案，下载聚类数组与参数。
5. 如需 LLM 建议，配置密钥后提交分析；人工类别与备注单独保存。

支持 MIB 的 `U08`、`U16`、`U32` 存储格式，以及 `1x1`、`2x2`、`2x2G` 探测器布局。暂不支持打包的 `R64`。扫描行数乘列数必须与帧数相等；配套 `.hdr` 应与 `.mib` 放在一起，缺少尺寸时可手动填写。

聚类编号只代表无监督分组。真空、晶体、非晶和混合类别来自 LLM 建议或人工确认，不按簇编号自动生成。

## 数据与结果

新导入的数据、分析结果和任务记录默认保存到 `workspace/`，与代码分开。已有数据库中的文件路径继续有效。

- `FOURDLLM_DATA_DIR`：修改数据和任务保存目录。
- `FOURDLLM_CONFIG`：指定数据库配置文件。
- `DATABASE_URI`：直接提供数据库连接，优先于配置文件。

任务采用串行队列和独立 Python 工作进程。刷新或关闭浏览器不会停止任务；取消请求在当前计算步骤或网络请求结束后的检查点生效。应用重启后继续处理排队任务，不从失败步骤自动恢复。

CIF 管理与模拟保留在 MCP 中，可通过 `uv sync --extra simulation` 安装模拟依赖。CIF 图案比较当前未实现，返回 `not_implemented`。

详见[完整文档](docs/documentation_zh.md)与 [CLI 说明](helper/CLI_TOOLS.md)。

## 许可与致谢

MIT，见 [LICENSE](LICENSE)。项目包含 [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp) 与 [ia-programming/mcp-images](https://github.com/ia-programming/mcp-images) 的工作，保留了 [LICENSES](LICENSES) 中的原始许可声明。
