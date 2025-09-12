# postgres_mcp.analyze

This package provides an asynchronous entrypoint to run the pre_class_v2 clustering + plotting
pipeline on scans already cataloged in the PostgreSQL database. It is designed to be called from
inside the MCP server code (or other Python programs) and follows the MCP sql_driver conventions.

Exported API

- analyze_scan(sql_driver, scan_identifier, out_root, k_clusters=16, seed=0, device=None)
  - sql_driver: MCP SQL driver (async interface) exposing execute_param_query(query, params)
  - scan_identifier: either integer scan_id or string scan_name
  - out_root: output directory where montages, xy maps and npz are written
  - k_clusters: k for kmeans
  - seed: random seed
  - device: torch device string (eg. 'cpu' or 'cuda') or None to auto-detect

Behavior

- The function queries `raw_mat_files` joined with `scans` to get ordered .mat file paths
  for the given scan identifier.
- The .mat files are loaded in background threads and stacked into an array (N, H, W).
- A synchronous clustering routine (feature extraction, PCA, kmeans, montage creation, XY map)
  is executed in a threadpool worker to avoid blocking the event loop.
- The function returns a dictionary with file paths for generated outputs.

Usage from MCP server
In a tool inside `postgres_mcp/server.py` you can call it like:

```python
from postgres_mcp.analyze import analyze_scan

# inside an async MCP tool function:
result = await analyze_scan(sql_driver, scan_identifier, out_root="/tmp/scan_outputs", k_clusters=16)
# return result to user (list of paths etc.)
```

Notes

- Assumes `raw_mat_files.file_path` contains server-local absolute paths to .mat files.
- The pipeline uses `pre_class_v2` code for feature extraction and plotting. Ensure dependencies (torch, numpy, scipy, matplotlib) are installed.
- Outputs are saved under `out_root/<scan_identifier>/...`.
