# 4DLLM - 4DSTEM 材料分析工具

面向4DSTEM的微观材料核心分类、数据分析与关系构建工具。

---

此项目使用了 [postgres-mcp](https://github.com/crystaldba/postgres-mcp), 原始许可证位于 [LICENSE-postgres-mcp](LICENSES/LICENSE-postgres-mcp)。

## 快速开始

### 1. 环境准备

确保已安装以下依赖：
- Python 3.13+
- PostgreSQL 17+
- Docker (推荐用于数据库)
- uv 包管理器

### 2. 数据库设置

```bash
# 启动数据库容器
docker-compose up -d

# 初始化数据库表
python init_enhanced_db.py
```

### 3. 配置文件

复制并编辑配置文件：
```bash
cp config/database.json.example config/database.json
cp config/api_keys.json.example config/api_keys.json
```

在 `config/database.json` 中填入数据库凭据，在 `config/api_keys.json` 中配置LLM API密钥。

## 核心功能

### 数据处理流程

1. **数据导入**: 将原始 .mib 文件转换为 .mat 格式并存入数据库
2. **聚类分析**: 使用 K-Means 对衍射图案进行无监督分类
3. **LLM分析**: 利用大语言模型对聚类结果进行语义分析
4. **结果存储**: 将分析结果持久化到数据库

### 命令行工具

```bash
# 导入数据
python main_pipeline_import.py /path/to/data.mib

# 聚类分析
python helper/analyze_scan_cli.py --scan-id 1 --k-clusters 16

# LLM分析
python enhanced_llm_analysis_pipeline.py scan_name
```

## 项目结构

```
4DLLM/
├── config/                 # 配置文件
├── helper/                 # 命令行工具
├── postgres_mcp/          # 数据库相关模块
├── api_manager/           # API密钥管理
├── docker/                # Docker配置
├── Data/                  # 处理后的数据文件
├── Raw/                   # 原始数据文件
└── llm_analysis_outputs/  # LLM分析结果
```

## 数据库设计

数据库包含以下核心表：
- `scans`: 扫描实验元数据
- `raw_mat_files`: 原始.mat文件引用
- `diffraction_patterns`: 基础衍射点数据
- `clustering_runs`: 聚类实验日志
- `identified_clusters`: 识别出的聚类信息
- `pattern_cluster_assignments`: 点到聚类的分配关系
- `llm_analyses`: LLM分析结果
- `llm_analysis_results`: 最终分析结果

## 开发指南

### 添加新功能

1. 创建新分支进行开发
2. 实现功能并添加相应测试
3. 更新文档
4. 提交Pull Request

### 数据库修改

如需修改数据库结构：
1. 更新 `enhanced_unified_database_schema.sql`
2. 修改 `init_enhanced_db.py` 中的SQL语句
3. 运行 `python init_enhanced_db.py` 重新初始化数据库

## 故障排除

### 常见问题

1. **数据库连接失败**: 检查Docker容器是否正在运行，确认数据库凭据正确
2. **LLM API错误**: 检查API密钥配置，确认网络连接正常
3. **内存不足**: 减少批处理大小或切换到CPU处理

### 获取帮助

如遇到问题，请查看相关文档或提交Issue。