# 4D-STEM Research Platform

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

### Command Line Tools

```bash
# Import data
python main_pipeline_import.py /path/to/data.mib

# Clustering analysis
python helper/analyze_scan_cli.py --scan-id 1 --k-clusters 16

# LLM analysis
python enhanced_llm_analysis_pipeline.py scan_name

# CIF management
python cif_analysis/cif_manager.py --download http://www.crystallography.net/cod/1572953.cif

# Pattern simulation
python cif_analysis/simulate_patterns.py --cif-id 1 --count 100

# Pattern comparison
python cif_analysis/compare_patterns.py --scan-id 1 --cif-id 1
```

## Project Structure

```
4DSTEM_Research_Platform/
├── config/                 # Configuration files
├── helper/                 # Command line tools
├── postgres_mcp/          # Database related modules
├── api_manager/           # API key management
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