import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from ..analyze.class_base import center_crop_cpu, preprocess_cpu_for_montage

CATEGORIES = {0: 'vacuum', 1: 'crystalline', 2: 'amorphous', 3: 'mixed'}


class PatternAnalyzer:
    def __init__(self, sql_driver, data_root=None):
        self.sql_driver = sql_driver
        self._temporary = TemporaryDirectory(prefix='4dllm-llm-')
        self.cluster_dir = Path(self._temporary.name)

    async def get_clustered_patterns(self, scan_name, cluster_id=None):
        condition = 'AND ic.id = %s' if cluster_id is not None else ''
        params = [scan_name] + ([cluster_id] if cluster_id is not None else [])
        rows = await self.sql_driver.execute_query(
            'SELECT dp.id AS pattern_id, rmf.row_index, dp.col_index, rmf.file_path, '
            's.id AS scan_id, s.scan_name, ic.id AS cluster_id, ic.cluster_index AS cluster_label, ic.run_id '
            'FROM pattern_cluster_assignments pca JOIN identified_clusters ic ON pca.cluster_id = ic.id '
            'JOIN clustering_runs cr ON ic.run_id = cr.id JOIN scans s ON cr.scan_id = s.id '
            'JOIN diffraction_patterns dp ON pca.pattern_id = dp.id '
            'JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id '
            f'WHERE s.scan_name = %s {condition} ORDER BY rmf.row_index, dp.col_index', params)
        return [row.cells for row in rows or []]

    async def get_all_clusters_for_scan(self, scan_name, clustering_run_id=None):
        condition = 'cr.id = %s' if clustering_run_id is not None else 'cr.id = (SELECT MAX(id) FROM clustering_runs WHERE scan_id = s.id)'
        params = [scan_name] + ([clustering_run_id] if clustering_run_id is not None else [])
        rows = await self.sql_driver.execute_query(
            'SELECT ic.id FROM identified_clusters ic JOIN clustering_runs cr ON ic.run_id = cr.id '
            f'JOIN scans s ON cr.scan_id = s.id WHERE s.scan_name = %s AND {condition} ORDER BY ic.cluster_index', params)
        return [row.cells['id'] for row in rows or []]

    def prepare_pattern_image(self, pattern_info):
        data = scipy.io.loadmat(pattern_info['file_path'])['data']
        image = preprocess_cpu_for_montage(center_crop_cpu(data[pattern_info['col_index'] - 1]))
        path = self.cluster_dir / f"pattern_{pattern_info['pattern_id']}.png"
        plt.imsave(path, image, cmap='gray', vmin=0, vmax=1)
        return str(path)

    async def save_analysis_results(self, cluster_id, analysis_results, representative_patterns=None, context=None):
        pairs = [(pattern, result['analysis']) for pattern, result in zip(representative_patterns or [], analysis_results)
                 if 'error' not in result['analysis']]
        if not pairs:
            raise ValueError('No successful LLM results')
        counts = Counter(int(result['classification_code']) for _, result in pairs)
        code = counts.most_common(1)[0][0]
        details = {'dominant_classification_code': code, 'samples': [result for _, result in pairs], **(context or {})}
        async with self.sql_driver.transaction() as tx:
            rows = await tx.execute_query(
                'INSERT INTO llm_analyses (cluster_id, representative_patterns_count, llm_assigned_class, llm_detailed_features) '
                'VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT (cluster_id) DO UPDATE SET '
                'representative_patterns_count = EXCLUDED.representative_patterns_count, '
                'llm_assigned_class = EXCLUDED.llm_assigned_class, llm_detailed_features = EXCLUDED.llm_detailed_features, '
                'analysis_timestamp = CURRENT_TIMESTAMP RETURNING id',
                [cluster_id, len(pairs), CATEGORIES[code], json.dumps(details)])
            analysis_id = rows[0].cells['id']
            await tx.execute_query('DELETE FROM llm_representative_patterns WHERE analysis_id = %s', [analysis_id])
            for pattern, result in pairs:
                await tx.execute_query(
                    'INSERT INTO llm_representative_patterns (analysis_id, pattern_id, selection_reason) VALUES (%s, %s, %s)',
                    [analysis_id, pattern['pattern_id'], 'Evenly spaced scan coordinates'])
                rows = await tx.execute_query(
                    'INSERT INTO llm_analysis_results (pattern_id, scan_id, clustering_run_id, cluster_id, llm_analysis_id, '
                    'row_index, col_index, cluster_index, llm_assigned_class, llm_detailed_features) '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) ON CONFLICT (pattern_id) DO UPDATE SET '
                    'clustering_run_id = EXCLUDED.clustering_run_id, cluster_id = EXCLUDED.cluster_id, '
                    'llm_analysis_id = EXCLUDED.llm_analysis_id, cluster_index = EXCLUDED.cluster_index, '
                    'llm_assigned_class = EXCLUDED.llm_assigned_class, llm_detailed_features = EXCLUDED.llm_detailed_features, '
                    'analysis_timestamp = CURRENT_TIMESTAMP RETURNING id',
                    [pattern['pattern_id'], pattern['scan_id'], pattern['run_id'], cluster_id, analysis_id,
                     pattern['row_index'], pattern['col_index'], pattern['cluster_label'],
                     CATEGORIES[int(result['classification_code'])], json.dumps(result)])
                result_id = rows[0].cells['id']
                await tx.execute_query('DELETE FROM llm_analysis_tags WHERE result_id = %s', [result_id])
                await tx.execute_many(
                    'INSERT INTO llm_analysis_tags (result_id, tag_category, tag_value) VALUES (%s, %s, %s)',
                    [(result_id, key[:100], str(value)[:100]) for key, value in result.items()
                     if key not in ('timestamp', 'description') and isinstance(value, (str, int, float))])
        return analysis_id

    def cleanup_temp_images(self):
        self._temporary.cleanup()
