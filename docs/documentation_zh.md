# 工作台与核心流程

## 运行方式

`./run.sh` 启动本地浏览器工作台；`uv run --locked python main.py` 启动 MCP。首次构建页面需要 npm，构建完成后只需 Python 环境和 PostgreSQL。

页面仅通过本机地址访问。配置位于 `config/database.json` 和 `config/api_keys.json`。可在设置页修改，密码和密钥不会通过读取配置接口返回；密码或密钥留空时保留已有值。`DATABASE_URI` 环境变量存在时优先使用它。

连接成功后会执行 `SQL/schema.sql`，创建缺失的表、字段、索引与约束，不删除现有表。手动初始化使用：

```bash
uv run --locked python setup_database.py --config config/database.json
```

## 导入

页面通过 Python 后端浏览设置中的数据目录，不需要把大型扫描文件经浏览器上传。目录必须存在；路径解析后仍须位于配置的目录内。

导入步骤：

1. 检查 MIB 格式、完整帧、扫描尺寸和扫描名称。
2. 把 MIB 与配套 HDR 复制到独立临时目录，同时计算 SHA-256。
3. 按实际扫描行数生成 MAT 文件，并保存总强度预览。
4. 在同一个数据库事务中写入扫描、MAT 引用和衍射点，衍射点使用 COPY 批量写入。
5. 成功后保留独立扫描目录；失败或取消时撤销数据库操作并移除本次临时数据。

重名扫描会拒绝导入，相同 SHA-256 的数据也会拒绝重复入库。旧版扫描没有摘要字段，无法自动识别它们与新文件是否为同一份数据。

支持 `U08`、`U16`、`U32`，不支持打包 `R64`。探测器布局支持 `1x1`、`2x2`、`2x2G`。扫描尺寸优先使用用户输入，其次 HDR 或可用的 MIB 头元数据；65536 帧且无尺寸时沿用 256×256。其他扫描缺少尺寸时要求填写，避免猜测。

目录导入处理当前目录内的 MIB，不递归。页面里的行列数可作为该批文件的统一尺寸。

## 聚类与结果

按行读取 MAT，逐批提取特征，再对特征矩阵进行 PCA 和 K-Means。原始数据不再整体转成 float32 并拼接。提取特征后的内存仍随图案数增长；工作台没有提供无限规模的流式 PCA。

每次运行使用独立目录，保存：

- `clustering.npz`：`labels`、`xs`、`ys`、到中心的 `distances`。扫描坐标从 1 开始。
- `parameters.json`：随机种子、设备选择、特征参数、PCA 与 K-Means 设置。
- `cluster_map.png`：中性的 Cluster 编号图。
- `montages/cluster_N.png`：每簇距离中心最近的代表图案。
- `individual_images/cluster_N/`：代表图案单图。

数据库中 `diffraction_patterns` 保存当前分配，`pattern_cluster_assignments` 保留各次运行的分配。写入分配和更新当前结果在同一事务中完成。GUI 和 MCP/CLI 的聚类使用同一个进程锁，避免同时占用计算设备。

## LLM 与人工标注

当前 LLM 接口使用 Gemini `generateContent` 格式，可以配置 Google 接口或兼容代理。每簇默认选择五个按扫描坐标均匀分布的样本，默认并发三路。

LLM 结果必须包含有效 `classification_code`：0 真空、1 晶体、2 非晶、3 混合。不合法或失败的响应记录为错误，不以簇编号补造结果。部分样本或簇失败时，任务报告保留错误并标明部分完成。

簇级 LLM 建议与样本结果写入数据库；模型、提示词和返回样本保留在分析详情与任务报告里。人工类别和备注写入独立的 `cluster_annotations` 表，不覆盖 LLM 建议。

GUI 的 LLM 按钮分析提交任务时解析到的运行；历史运行可以单独查看。多次对同一簇进行 LLM 分析时，数据库显示最新建议，任务报告保留之前的返回结果。

## 任务

任务记录和日志位于 `workspace/jobs/`。队列串行启动独立 Python 工作进程。状态为 `queued`、`running`、`succeeded`、`failed` 或 `cancelled`。

取消采用协作式检查：复制数据、转换行、提取特征、保存代表图与批量入库时有检查点；PCA、K-Means 与 LLM 请求结束后才响应取消。取消已经提交完成的操作不会撤销它，任务仍报告成功。LLM 的各簇结果分别提交，取消或失败前已完成的簇建议保留。

正常关闭应用会等待当前工作进程完成，剩余排队任务留到下次启动；重启后继续排队任务。工作进程异常退出的任务记为失败，点击重试会创建新任务。

## HTTP 接口

浏览器和 HTTP 接口使用同一来源。以下路径均带 `/api` 前缀。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/status`、`/settings` | 连接状态、可展示配置 |
| PUT | `/settings` | 保存数据库、LLM 和导入目录 |
| GET | `/files?path=...` | 浏览数据目录 |
| POST | `/inspect` | 检查 MIB，输入 `path` 与可选 `scan_shape` |
| POST | `/imports` | 导入单文件，输入 `path`、可选 `scan_shape` 与 `scan_name` |
| POST | `/import-folders` | 导入目录，输入 `path` 与可选 `scan_shape` |
| GET | `/scans`、`/scans/{id}` | 扫描列表和分析历史 |
| GET | `/scans/{id}/overview`、`/scans/{id}/pattern?row=1&col=1` | 预览图 |
| POST | `/scans/{id}/cluster` | 提交聚类，输入 `k`、`seed`、`device` |
| POST | `/scans/{id}/llm` | 提交 LLM，输入可选 `cluster_id`、`samples`、`concurrency` |
| GET | `/runs/{id}`、`/runs/{id}/grid` | 簇结果和空间图数据 |
| GET | `/runs/{id}/files/{name}` | 下载结果文件 |
| PUT | `/clusters/{id}/annotation` | 保存人工类别和备注；`category: null` 清除标注 |
| GET | `/jobs`、`/jobs/{id}` | 任务列表和状态 |
| POST | `/jobs/{id}/cancel`、`/jobs/{id}/retry` | 取消或重新提交 |
| GET | `/jobs/{id}/log`、`/jobs/{id}/report` | 日志、完整任务报告 |

## MCP 与可选 CIF

原有扫描导入、聚类、图像查看、LLM 和 CIF 管理入口保留。`ingest_scan_from_mib` 增加可选的 `scan_rows`、`scan_columns`、`scan_name`；`check_analysis_status` 返回浏览器任务记录。

未启用的数据库索引调优、健康检查、Explain 和 top-query 模块已删除；原有 SQL 执行与只读 SQL 校验保留。MCP 的同步调用不会加入浏览器队列，但聚类共用核心锁。

CIF 模拟需要 `uv sync --extra simulation`。CIF 图案比较尚未实现，MCP 返回 `not_implemented`；GUI 第一版提供扫描、聚类和 LLM 工作流。
