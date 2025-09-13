# LLM Analysis for 4D-STEM Diffraction Patterns

## Overview

The LLM analysis module provides automated analysis of clustered 4D-STEM diffraction patterns using a generic LLM API. This system can identify phases, structural features, and provide detailed characterization of diffraction patterns.

## Features

- **Parallel Processing**: Analyze multiple patterns simultaneously using async batch processing
- **Smart Pattern Selection**: Automatically select representative patterns from large clusters
- **Phase Identification**: Use AI to identify crystalline phases and material types
- **Structured Analysis**: Store results in database with structured metadata
- **Rate Limiting**: Built-in API rate limiting and key rotation for reliability

## Prerequisites

1. **Database Setup**: Ensure the database schema includes LLM analysis tables (should already be included)
2. **API Keys**: Obtain API keys for your LLM service and configure them
3. **Data**: Have clustered diffraction pattern data available in the database

## Configuration

### API Keys Setup

Create or update `/config/api_keys.json` with your LLM API keys:

```json
{
  "api_keys": [
    "your-api-key-1",
    "your-api-key-2"
  ]
}
```

Multiple API keys are recommended for:

- Better rate limiting tolerance
- Fault tolerance if one key fails
- Improved parallel processing performance

## MCP Tools Available

### 1. `run_llm_cluster_analysis`

Runs complete LLM analysis on clustered diffraction patterns.

**Parameters:**

- `scan_name` (required): Name of the scan to analyze
- `cluster_id` (optional): Specific cluster ID (if not provided, analyzes all clusters)
- `max_patterns_per_cluster` (default: 5): Maximum patterns to analyze per cluster
- `batch_size` (default: 3): Number of patterns to process in parallel

**Example:**

```
Run LLM analysis on scan "2" for all clusters
```

### 2. `get_llm_analysis_summary`

Get a summary of all LLM analyses performed on a scan.

**Parameters:**

- `scan_name` (required): Name of the scan

**Example:**

```
Get analysis summary for scan "2"
```

### 3. `get_cluster_llm_details`

Get detailed LLM analysis results for a specific cluster.

**Parameters:**

- `scan_name` (required): Name of the scan
- `cluster_id` (required): ID of the cluster

**Example:**

```
Get detailed analysis for cluster 1 in scan "2"
```

## Analysis Pipeline

The analysis follows these steps:

1. **Pattern Retrieval**: Get clustered patterns from database
2. **Representative Selection**: Select diverse patterns if cluster is large
3. **Image Preparation**: Convert .mat files to PNG images for LLM analysis
4. **Batch Processing**: Process multiple images in parallel using async operations
5. **LLM Analysis**: Send images to LLM API with structured analysis prompt
6. **Result Storage**: Save analysis results and metadata to database
7. **Cleanup**: Remove temporary image files

## Analysis Output

The LLM provides structured analysis including:

- **Phase Type**: Identified crystalline phase or material type
- **Structural Features**: Description of diffraction features (spots, rings, streaks)
- **Symmetry**: Observations about pattern symmetry
- **Quality Assessment**: Pattern quality rating (clear/noisy/damaged)
- **Notable Features**: Special characteristics or anomalies
- **Confidence**: Analysis confidence level

## Database Schema

The analysis uses these main tables:

- `llm_analyses`: Main analysis results per cluster
- `llm_representative_patterns`: Patterns selected for analysis
- `llm_analysis_tags`: Structured tags for querying

## Error Handling

The system includes robust error handling:

- **API Failures**: Automatic retry with different API keys
- **Rate Limiting**: Built-in delays and backoff strategies
- **Image Processing**: Graceful handling of corrupted .mat files
- **Database Errors**: Transaction rollback and error logging

## Performance Considerations

- **Batch Size**: Adjust based on API rate limits (default: 3)
- **Pattern Limits**: Large clusters are subsampled to avoid excessive API calls
- **Concurrent Processing**: Uses async operations for better throughput
- **Memory Management**: Temporary images are cleaned up automatically

## Troubleshooting

### Common Issues:

1. **"No API keys found"**

   - Ensure `/config/api_keys.json` exists with valid `api_keys` array

2. **"No patterns found for cluster"**

   - Verify scan exists and has clustered data
   - Check cluster_id is valid

3. **API Rate Limit Errors**

   - Add more API keys to configuration
   - Reduce batch_size parameter
   - Check API key quotas

4. **Image Processing Errors**
   - Verify .mat files exist in Data directory
   - Check file permissions
   - Ensure scan directory structure is correct

## Example Workflow

1. First, check what scans are available:

   ```
   SELECT DISTINCT scan_name FROM scans;
   ```

2. Run analysis on a scan:

   ```
   Use run_llm_cluster_analysis with scan_name="2"
   ```

3. Check results:

   ```
   Use get_llm_analysis_summary with scan_name="2"
   ```

4. Get detailed results for specific clusters:
   ```
   Use get_cluster_llm_details with scan_name="2" and cluster_id=1
   ```

## Technical Details

- **Framework**: Uses MCP (Model Context Protocol) server architecture
- **AI Model**: Generic LLM for image analysis
- **Database**: PostgreSQL with JSONB storage for flexible metadata
- **Image Processing**: scipy.io for .mat files, matplotlib for PNG conversion
- **Async Framework**: aiohttp for parallel API calls
- **Rate Limiting**: Token bucket algorithm with configurable limits

This implementation provides a complete, production-ready solution for automated 4D-STEM pattern analysis using large language models.
