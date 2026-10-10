# Analysis core

`analyze_scan(sql_driver, scan_identifier, out_root=None, k_clusters=16, seed=0, device=None, progress=None)` resolves a scan, loads MAT rows sequentially, extracts polar features, performs PCA and K-Means, writes a new result directory, and stores all assignments in one transaction.

`progress(stage, fraction, message)` is optional. Raising from it stops processing at a checkpoint and triggers cleanup. A shared process lock prevents overlapping clustering from GUI, CLI, and MCP.

`class_base.py` owns the algorithms shared with the interactive `pre_class_v2.py` entry point. `db_analyze.py` owns orchestration, result persistence, and rendering. Clustering maps show neutral cluster IDs; LLM suggestions and human labels are separate.

Current labels live in `diffraction_patterns`. Historical assignments remain in `pattern_cluster_assignments`. `regenerate_classification_map` reads assignments for the requested historical run.

See [documentation](../../docs/documentation.md) for outputs and task behavior.
