[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL Version](https://img.shields.io/badge/PostgreSQL-17+-336791.svg)](https://www.postgresql.org/download/)
[![Docker](https://img.shields.io/badge/Docker-✓-1D63ED.svg)](https://www.docker.com/)
[![UV](https://img.shields.io/badge/uv-✓-de5fe9.svg)](https://docs.astral.sh/uv/) 
[![Model Context Protocol](https://img.shields.io/badge/MCP-Protocol-eeeeee.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/github/license/Frank0415/4DLLM
)](https://opensource.org/licenses/MIT)

# 4DLLM

一个基于 MCP 协议的自动化 4DSTEM 显微材料分析平台。

## 概述

本平台为4D-STEM（四维扫描透射电子显微镜）数据分析提供完整解决方案，包括：

1. **K-Means聚类** 用于无监督模式分类
2. **大语言模型（LLM）分析** 用于语义解释
3. **CIF模拟与对比** 用于晶体结构识别
4. **数据库存储** 具备完整可追溯性和查询能力

系统基于PostgreSQL数据库，拥有支持整个4D-STEM研究工作流程的全面架构。

## 数据库结构

### 核心表（共14个）
1. **scans** - 扫描实验元数据
2. **raw_mat_files** - 原始.mat文件引用
3. **diffraction_patterns** - 基础衍射点数据
4. **clustering_runs** - K-Means聚类实验日志
5. **identified_clusters** - 已识别的聚类信息
6. **pattern_cluster_assignments** - 点到聚类的分配
7. **llm_analyses** - LLM聚类分析结果
8. **llm_representative_patterns** - 代表性模式选择
9. **llm_analysis_results** - 最终扁平化分析结果
10. **llm_analysis_tags** - 结构化分析标签
11. **llm_analysis_batches** - 批处理元数据
12. **cif_files** - CIF文件信息和晶体学数据
13. **simulated_patterns** - 从CIF文件生成的模拟衍射图案
14. **pattern_comparisons** - 实验和模拟图案的对比结果

### 视图（共7个）
1. **cluster_statistics** - 聚类分布统计
2. **spatial_cluster_distribution** - 空间聚类映射
3. **llm_analysis_overview** - LLM分析摘要
4. **tag_statistics** - 结构化标签分析
5. **batch_processing_stats** - 批处理统计
6. **cif_statistics** - CIF文件统计
7. **comparison_overview** - 图案对比结果

## 快速入门

### 1. 系统要求
- Python 3.13+
- PostgreSQL 17+
- Docker（推荐用于数据库）
- uv 包管理器

### 2. 数据库设置
```bash
# 启动数据库容器
docker-compose -f docker/docker-compose.yml up -d

# 初始化数据库结构
python setup_database.py
```

### 3. 配置文件
复制并编辑配置文件：
```bash
cp config/db_config_example.json config/database.json
cp config/api_keys_example.json config/api_keys.json
```

在 `config/database.json` 中填写数据库凭证，在 `config/api_keys.json` 中配置LLM API密钥。

## 核心功能

### 数据处理流程

1. **数据导入**：将原始.mib文件转换为.mat格式并存入数据库
2. **聚类分析**：使用K-Means对衍射图案进行无监督分类
3. **LLM分析**：使用大语言模型对聚类结果进行语义分析
4. **CIF模拟**：从晶体结构数据库生成理论图案
5. **图案对比**：对比实验和模拟图案进行识别
6. **结果存储**：将所有分析结果持久化到数据库

### 可用的工具

#### 1. 数据库管理工具

| 工具 | 描述 |
|:------|:------|
| **`list_schemas`** | 列出数据库中的所有模式(schema) |
| **`list_objects`** | 列出指定模式中的对象 |
| **`get_object_details`** | 显示特定数据库对象的详细信息 |
| **`execute_sql`** | 执行任意 SQL 查询 |

#### 2. 扫描分析工具

| 工具 | 描述 |
|:------|:------|
| **`analyze_scan_tool`** | 对 4D-STEM 扫描运行聚类分析 |
| **`display_classification_results`** | 显示已完成分析的扫描的分类结果 |
| **`verify_cluster_storage_tool`** | 验证指定扫描的聚类分配是否存储在数据库中 |
| **`regenerate_classification_map_tool`** | 使用 LLM 分配的类别重新生成分类图 |
| **`show_classification_map`** | 通过本地文件 URL 显示分类图 |
| **`list_classification_images`** | 列出扫描中所有可用的分类图像和映射 |
| **`show_classification_overview`** | 显示扫描的全面分类概览(包含对应的映射和样品蒙太奇) |

#### 3. 扫描导入与信息检索工具

| 工具 | 描述 |
|:------|:------|
| **`ingest_scan_from_mib`** | 处理 .mib 文件，解压为多个 .mat 文件，并将扫描信息录入目录数据库 |
| **`list_ingested_scans`** | 列出所有已导入的科学扫描的高级摘要 |
| **`get_scan_details`** | 获取指定扫描中的原始 .mat 文件的详细列表 |

#### 4. 原始图像处理工具

| 工具 | 描述 |
|:------|:------|
| **`show_raw_image`** | 根据图像所在的 .mat 文件、图像编号和扫描 ID 显示原始图像，并附带 LLM 分析标签 |

#### 5. 聚类分析与 LLM 工具

| 工具 | 描述 |
|:------|:------|
| **`generate_cluster_consensus_tool`** | 利用 LLM 为一次扫描中的所有聚类生成共识描述 |
| **`show_cluster_consensus`** | 显示一次扫描中特定聚类的共识描述 |
| **`show_cluster_montages`** | 显示一次扫描中特定聚类的代表性衍射图案蒙太奇 |
| **`run_llm_cluster_analysis`** | 对一次特定扫描中经过聚类的衍射图样运行基于 LLM 的分析 |
| **`get_llm_analysis_summary`** | 获取一次特定扫描上所有 LLM 分析的摘要 |
| **`get_cluster_llm_details`** | 获取一个特定聚类的详细 LLM 分析结果 |

#### 6. CIF 文件处理工具

| 工具 | 描述 |
|:------|:------|
| **`download_cif_file`** | 从晶体数据库下载 CIF 文件并存储到数据库 |
| **`upload_cif_file`** | 上传本地 CIF 文件到数据库 |
| **`generate_simulated_patterns`** | 从 CIF 文件生成模拟的衍射图案 |
| **`compare_patterns`** | 比较实验衍射图案与从 CIF 文件生成的模拟图案 |

#### 7. LLM 连接性与测试工具

| 工具 | 描述 |
|:------|:------|
| **`test_llm_analysis`** | 通过分析本地图像文件测试 LLM 连接性 |


## 项目结构

```
4DLLM/
├── config/                 # 配置文件
├── helper/                 # 命令行工具
├── postgres_mcp/           # 数据库相关模块
├── api_manager/            # API密钥管理
├── docker/                 # Docker配置
├── cif_analysis/           # CIF分析模块
├── SQL/                    # 数据库架构
├── Data/                   # 处理后的数据文件
├── Raw/                    # 原始数据文件
└── docs/                   # 文档
```

## 数据库设计原则

### 规范化结构
数据库遵循规范化原则，消除冗余并确保数据完整性：
- 每个实体都有单一的信息来源
- 通过外键约束强制执行关系
- 级联删除维护引用完整性

### 分层架构
架构按逻辑层组织：
1. **原始数据层** - 原始扫描数据存储
2. **聚类分析层** - K-Means处理结果
3. **LLM分析层** - 聚类的语义解释
4. **LLM结果层** - 扁平化的分析结果供查询
5. **CIF模拟层** - 晶体结构模拟和对比

### 为分析优化
战略性索引和视图实现实效查询：
- 空间索引用于基于坐标的查询
- 复合索引用于常见过滤器组合
- 预计算视图用于频繁的分析模式

## 高级特性

### 限速LLM处理
系统实现智能限速以防止压垮LLM API：
- 可配置的并发限制（默认：64个并行任务）
- 带指数退避的自动重试
- 批处理以提高吞吐量

### 图案对比框架
高级图案对比能力：
- 多种相似性度量（SSIM、MSE、余弦相似性等）
- 实验和模拟图案之间的批量对比
- 自动识别的置信度评分

### 可扩展标记系统
结构化标记系统实现灵活分类：
- 领域特定类别（phase_type、crystallinity_level等）
- 置信度评分用于不确定性量化
- 统计视图用于标记分析

## 开源许可与致谢

### 核心项目
- **4DLLM**: 本项目已在 GitHub 上开源。
  - **代码仓库**: [https://github.com/Frank0415/4DLLM](https://github.com/Frank0415/4DLLM)
  - **许可证**: [MIT License](https://opensource.org/licenses/MIT)

### 依赖的开源组件
本项目的开发建立在以下优秀开源项目的基础之上：

1.  **MCP 服务器框架**:
    - 基于 [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp) 构建。
    - **许可证**: [MIT License](https://opensource.org/licenses/MIT)

2.  **图像处理功能**:
    - 图片读取功能基于 [ia-programming/mcp-images](https://github.com/ia-programming/mcp-images)。
    - **许可证**: [MIT License](https://opensource.org/licenses/MIT)

### 相关项目
- **4DLLM-arxiv-mcp-server**: 我们用于ArXiv论文分析的MCP服务器也已开源。
  - **代码仓库**: [https://github.com/Frank0415/4DLLM-arxiv-mcp-server](https://github.com/Frank0415/4DLLM-arxiv-mcp-server)
  - **基于项目**: [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server)
  - **许可证**: [Apache-2.0 license](https://opensource.org/licenses/Apache-2.0)

我们衷心感谢以上所有项目和贡献者。

## 故障排除

### 常见问题

1. **数据库连接失败**：检查Docker容器是否运行并验证数据库凭证
2. **LLM API错误**：检查API密钥配置并确保网络连接
3. **内存不足**：减少批处理大小或切换到CPU处理
4. **缺少依赖**：确保所有必需的Python包都已安装

### 获取帮助

对于本文档未涵盖的问题，请查看日志了解详细错误信息并查阅相关模块文档。

---
*文档最后更新：2025年9月*