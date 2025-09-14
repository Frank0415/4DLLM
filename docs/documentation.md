[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL Version](https://img.shields.io/badge/PostgreSQL-17+-336791.svg)](https://www.postgresql.org/download/)
[![Docker](https://img.shields.io/badge/Docker-✓-1D63ED.svg)](https://www.docker.com/)
[![UV](https://img.shields.io/badge/uv-✓-de5fe9.svg)](https://docs.astral.sh/uv/) 
[![Model Context Protocol](https://img.shields.io/badge/MCP-Protocol-eeeeee.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/github/license/Frank0415/4DLLM
)](https://opensource.org/licenses/MIT)

# 4DLLM

An automated 4DSTEM microscopic material analysis based on MCP.

## Overview

This platform provides a complete solution for analyzing 4D-STEM (Four-Dimensional Scanning Transmission Electron Microscopy) data through:

1. **K-Means Clustering** for unsupervised pattern classification
2. **Large Language Model (LLM) Analysis** for semantic interpretation
3. **CIF Simulation and Comparison** for crystal structure identification
4. **Database Storage** with full traceability and querying capabilities

The system is built on a PostgreSQL database with a comprehensive schema supporting the entire 4D-STEM research workflow.

## Database Structure

### Core Tables (14 total)
1. **scans** - Scan experiment metadata
2. **raw_mat_files** - Raw .mat file references
3. **diffraction_patterns** - Basic diffraction point data
4. **clustering_runs** - K-Means clustering experiment logs
5. **identified_clusters** - Identified cluster information
6. **pattern_cluster_assignments** - Point-to-cluster assignments
7. **llm_analyses** - LLM analysis results for clusters
8. **llm_representative_patterns** - Representative pattern selections
9. **llm_analysis_results** - Final flattened analysis results
10. **llm_analysis_tags** - Structured analysis tags
11. **llm_analysis_batches** - Batch processing metadata
12. **cif_files** - CIF file information and crystallographic data
13. **simulated_patterns** - Simulated diffraction patterns from CIF files
14. **pattern_comparisons** - Comparison results between experimental and simulated patterns

### Views (7 total)
1. **cluster_statistics** - Cluster distribution statistics
2. **spatial_cluster_distribution** - Spatial cluster mappings
3. **llm_analysis_overview** - LLM analysis summaries
4. **tag_statistics** - Structured tag analytics
5. **batch_processing_stats** - Batch processing metrics
6. **cif_statistics** - CIF file statistics
7. **comparison_overview** - Pattern comparison results

## Getting Started

### 1. Prerequisites
- Python 3.13+
- PostgreSQL 17+
- Docker (recommended for database)
- uv package manager

### 2. Database Setup
```bash
# Start database containers
docker-compose -f docker/docker-compose.yml up -d

# Initialize database with schema
python setup_database.py
```

### 3. Configuration
Copy and edit configuration files:
```bash
cp config/db_config_example.json config/database.json
cp config/api_keys_example.json config/api_keys.json
```

Fill in database credentials in `config/database.json` and configure LLM API keys in `config/api_keys.json`.

## Core Functionality

### Data Processing Workflow

1. **Data Import**: Convert raw .mib files to .mat format and store in database
2. **Clustering Analysis**: Use K-Means for unsupervised classification of diffraction patterns
3. **LLM Analysis**: Semantic analysis of clustering results using large language models
4. **CIF Simulation**: Generate theoretical patterns from crystal structure databases
5. **Pattern Comparison**: Compare experimental and simulated patterns for identification
6. **Result Storage**: Persist all analysis results to database

### Available Tools

#### 1. Database Management Tools

| Tool | Description |
|:-----|:------------|
| **`list_schemas`** | Lists all schemas in the database |
| **`list_objects`** | Lists objects in a specified schema |
| **`get_object_details`** | Shows detailed information about a specific database object |
| **`execute_sql`** | Executes any SQL query |

#### 2. Scan Analysis Tools

| Tool | Description |
|:-----|:------------|
| **`analyze_scan_tool`** | Runs clustering analysis on a 4D-STEM scan |
| **`display_classification_results`** | Displays classification results for a scan |
| **`verify_cluster_storage_tool`** | Verifies cluster assignments are stored for a scan |
| **`regenerate_classification_map_tool`** | Regenerates the classification map using LLM-assigned classes |
| **`show_classification_map`** | Displays a classification map from a file URL |
| **`list_classification_images`** | Lists all classification images and maps for a scan |
| **`show_classification_overview`** | Shows a comprehensive classification overview for a scan (including maps and sample montages) |

#### 3. Scan Ingestion and Information Retrieval Tools

| Tool | Description |
|:-----|:------------|
| **`ingest_scan_from_mib`** | Processes a .mib file, unpacks into .mat files, and catalogs the scan |
| **`list_ingested_scans`** | Lists a high-level summary of all ingested scientific scans |
| **`get_scan_details`** | Retrieves detailed information about the raw .mat files in a scan |

#### 4. Raw Image Handling Tools

| Tool | Description |
|:-----|:------------|
| **`show_raw_image`** | Displays a raw image from the database (identified by its image_in_mat, mat_number, and scan_id) along with LLM analysis tags |

#### 5. Cluster Analysis and LLM Tools

| Tool | Description |
|:-----|:------------|
| **`generate_cluster_consensus_tool`** | Generates consensus descriptions for all clusters in a scan using LLM |
| **`show_cluster_consensus`** | Displays a consensus description for a specific cluster in a scan |
| **`show_cluster_montages`** | Displays representative diffraction patterns for a specific cluster in a scan |
| **`run_llm_cluster_analysis`** | Runs LLM-based analysis on clustered diffraction patterns for a specific scan |
| **`get_llm_analysis_summary`** | Gets a summary of all LLM analyses performed on a scan |
| **`get_cluster_llm_details`** | Gets detailed LLM analysis results for a specific cluster |

#### 6. CIF File Handling Tools

| Tool | Description |
|:-----|:------------|
| **`download_cif_file`** | Downloads a CIF file from a crystallography database and stores it in the database |
| **`upload_cif_file`** | Uploads a local CIF file to the database |
| **`generate_simulated_patterns`** | Generates simulated diffraction patterns from a CIF file |
| **`compare_patterns`** | Compares experimental diffraction patterns with simulated patterns from a CIF file |

#### 7. LLM Connectivity and Testing Tools

| Tool | Description |
|:-----|:------------|
| **`test_llm_analysis`** | Tests LLM connectivity by analyzing a local image file with the configured API. |

## Project Structure

```
4DLLM/
├── config/                 # Configuration files
├── helper/                 # Command line tools
├── postgres_mcp/           # Database related modules
├── api_manager/            # API key management
├── docker/                 # Docker configuration
├── cif_analysis/           # CIF analysis modules
├── SQL/                    # Database schema
├── Data/                   # Processed data files
├── Raw/                    # Raw data files
└── docs/                   # Documentation
```

## Database Design Principles

### Normalized Structure
The database follows normalization principles to eliminate redundancy and ensure data integrity:
- Each entity has a single source of truth
- Relationships are enforced through foreign key constraints
- Cascading deletes maintain referential integrity

### Layered Architecture
The schema is organized in logical layers:
1. **Raw Data Layer** - Original scan data storage
2. **Clustering Analysis Layer** - K-Means processing results
3. **LLM Analysis Layer** - Semantic interpretation of clusters
4. **LLM Results Layer** - Flattened analysis results for querying
5. **CIF Simulation Layer** - Crystal structure simulation and comparison

### Optimized for Analytics
Strategic indexes and views enable efficient querying:
- Spatial indexes for coordinate-based queries
- Composite indexes for common filter combinations
- Pre-computed views for frequent analytical patterns

## Advanced Features

### Rate-Limited LLM Processing
The system implements intelligent rate limiting to prevent overwhelming LLM APIs:
- Configurable concurrency limits (default: 64 parallel tasks)
- Automatic retry with exponential backoff
- Batch processing for improved throughput

### Pattern Comparison Framework
Advanced pattern comparison capabilities:
- Multiple similarity metrics (SSIM, MSE, cosine similarity, etc.)
- Batch comparison between experimental and simulated patterns
- Confidence scoring for automated identification

### Extensible Tagging System
Structured tagging system for flexible categorization:
- Domain-specific categories (phase_type, crystallinity_level, etc.)
- Confidence scores for uncertainty quantification
- Statistical views for tag analysis

## Open Source Licensing & Acknowledgments

### Core Project
- **4DLLM**: This project is open-sourced on GitHub.
  - **Repository**: [https://github.com/Frank0415/4DLLM](https://github.com/Frank0415/4DLLM)
  - **License**: [MIT License](https://opensource.org/licenses/MIT)

### Dependent Open-Source Components
This project is built upon the following excellent open-source projects:

1.  **MCP Server Framework**:
    - Based on [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp).
    - **License**: [MIT License](https://opensource.org/licenses/MIT)

2.  **Image Handling Functionality**:
    - Image reading capabilities are based on [ia-programming/mcp-images](https://github.com/ia-programming/mcp-images).
    - **License**: [MIT License](https://opensource.org/licenses/MIT)

### Related Projects
- **4DLLM-arxiv-mcp-server**: Our MCP server for ArXiv paper analysis is also open-sourced.
  - **Repository**: [https://github.com/Frank0415/4DLLM-arxiv-mcp-server](https://github.com/Frank0415/4DLLM-arxiv-mcp-server)
  - **Based on**: [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server)
  - **License**: [Apache-2.0 license](https://opensource.org/licenses/Apache-2.0)

We extend our sincere gratitude to all the above projects and their contributors.

## Troubleshooting

### Common Issues

1. **Database Connection Failed**: Check if Docker containers are running and verify database credentials
2. **LLM API Errors**: Check API key configuration and ensure network connectivity
3. **Insufficient Memory**: Reduce batch size or switch to CPU processing
4. **Missing Dependencies**: Ensure all required Python packages are installed

### Getting Help

For issues not covered in this documentation, please check the logs for detailed error messages and consult the relevant module documentation.

---
*Documentation last updated: September 2025*