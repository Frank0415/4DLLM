# Concurrency Control for Analysis Tasks

This document explains how the concurrency control mechanism works for resource-intensive analysis tasks in the 4DLLM system.

## Problem

The 4D-STEM analysis tasks are resource-intensive and can take a long time to complete. Running multiple analysis tasks simultaneously can:

1. Overload system resources (CPU, memory, disk I/O)
2. Cause performance degradation or system instability
3. Lead to resource contention and unpredictable behavior

## Solution

We implement a file-based locking mechanism to ensure only one analysis task runs at a time.

## How It Works

### Lock File

The system uses a lock file located at `/tmp/4dllm_analysis.lock` to coordinate access to analysis resources.

### Lock Acquisition

1. When an analysis task is requested, the system first checks if the lock file exists
2. If the lock file exists, it checks if the lock is stale (older than 1 hour)
3. If the lock is stale, it cleans up the lock file and proceeds
4. If the lock is fresh, the new request is denied
5. If no lock exists, a new lock file is created atomically

### Lock Information

The lock file contains JSON information about the running task:
```json
{
  "job_id": "unique-job-identifier",
  "scan_identifier": "scan-name-or-id",
  "timestamp": 1234567890.123
}
```

### Lock Release

The lock is automatically released when:
1. The analysis task completes successfully
2. The analysis task encounters an error
3. The process is terminated
4. The lock is detected as stale (older than 1 hour)

## Tools

### analyze_scan_tool

The main analysis tool that performs feature extraction, PCA, k-means clustering, and generates visualization outputs. It runs synchronously but implements locking to prevent concurrent execution.

### check_analysis_status *(Deprecated)*

This tool was used for background processes but is no longer needed since we're running synchronously.

### get_analysis_log *(Deprecated)*

This tool was used for retrieving detailed logs from background processes but is no longer needed.

## Benefits

1. **Resource Protection**: Prevents system overload from multiple concurrent analysis tasks
2. **User Experience**: Provides clear error messages when tasks are denied
3. **Reliability**: Handles stale locks gracefully to prevent permanent blocking
4. **Simplicity**: Uses file-based locking which doesn't require additional infrastructure

## Error Handling

- If a task fails to start, the lock is automatically released
- Stale locks (older than 1 hour) are automatically detected and cleaned up
- Clear error messages are provided when tasks are denied due to existing locks

## Note on Background Processing

Initially, we attempted to run analysis tasks in the background using subprocesses. However, we found that the MCP framework terminates subprocesses when the tool function completes. Therefore, we reverted to a synchronous approach while maintaining the locking mechanism to prevent concurrent executions.