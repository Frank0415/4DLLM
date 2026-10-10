import asyncio
from datetime import datetime, timezone

import numpy as np

from .llm_orchestrator import LLMOrchestrator
from .pattern_analyzer import PatternAnalyzer

ANALYSIS_PROMPT = '''Analyze this electron diffraction pattern as a materials scientist.
Return a JSON object with classification_code (integer: 0 vacuum, 1 crystalline,
2 amorphous, 3 mixed), description (observed evidence), phase_type, quality,
structural_features, and confidence (high, medium, or low). Separate observations
from hypotheses. Do not infer a specific composition from appearance alone.'''


class AnalysisPipeline:
    def __init__(self, sql_driver, api_keys, base_url='https://generativelanguage.googleapis.com', model='gemini-2.5-pro', data_root=None):
        self.sql_driver = sql_driver
        self.orchestrator = LLMOrchestrator(api_keys, base_url, model)
        self.analyzer = PatternAnalyzer(sql_driver, data_root)

    async def analyze_scan_clusters(self, scan_name, cluster_id=None, max_patterns_per_cluster=5, batch_size=3, progress=None, max_total_patterns=None, clustering_run_id=None):
        started = datetime.now(timezone.utc)
        clusters = [cluster_id] if cluster_id is not None else await self.analyzer.get_all_clusters_for_scan(scan_name, clustering_run_id)
        result = {'status': 'success', 'scan_name': scan_name, 'clusters_analyzed': 0,
                  'total_patterns_analyzed': 0, 'cluster_results': {}, 'errors': [],
                  'model': self.orchestrator.model, 'prompt': ANALYSIS_PROMPT, 'start_time': started.isoformat()}
        try:
            if not clusters:
                raise ValueError('Run clustering before LLM analysis')
            if max_patterns_per_cluster < 1 or batch_size < 1:
                raise ValueError('Pattern count and concurrency must be positive')
            for index, identifier in enumerate(clusters):
                if progress:
                    progress('llm', index / len(clusters), f'Analyzing cluster {index + 1}/{len(clusters)}')
                remaining = max_patterns_per_cluster
                if max_total_patterns is not None:
                    remaining = min(remaining, max_total_patterns - result['total_patterns_analyzed'])
                    if remaining <= 0:
                        break
                single = await self._analyze_single_cluster(scan_name, identifier, remaining, batch_size)
                result['cluster_results'][str(identifier)] = single
                result['total_patterns_analyzed'] += single.get('patterns_analyzed', 0)
                if single['status'] == 'success':
                    result['clusters_analyzed'] += 1
                    result['errors'].extend(single.get('errors', []))
                else:
                    result['errors'].append(single['message'])
            if result['errors']:
                result['status'] = 'partial' if result['clusters_analyzed'] else 'error'
                result['message'] = '; '.join(result['errors'])
            result['duration_seconds'] = (datetime.now(timezone.utc) - started).total_seconds()
            return result
        finally:
            self.analyzer.cleanup_temp_images()

    async def _analyze_single_cluster(self, scan_name, cluster_id, max_patterns_per_cluster, batch_size):
        responses = []
        try:
            patterns = await self.analyzer.get_clustered_patterns(scan_name, cluster_id)
            if not patterns:
                raise ValueError(f'No patterns found for cluster {cluster_id}')
            indices = np.linspace(0, len(patterns) - 1, min(max_patterns_per_cluster, len(patterns)), dtype=int)
            selected = [patterns[int(index)] for index in indices]
            images = [await asyncio.to_thread(self.analyzer.prepare_pattern_image, pattern) for pattern in selected]
            responses = await self.orchestrator.analyze_batch(images, ANALYSIS_PROMPT, asyncio.Semaphore(batch_size))
            for response in responses:
                if 'error' not in response and (type(response.get('classification_code')) is not int or response['classification_code'] not in (0, 1, 2, 3)):
                    response['error'] = 'Invalid classification_code in LLM response'
            successful = [response for response in responses if 'error' not in response]
            if not successful:
                raise ValueError('; '.join(response.get('error', 'LLM failed') for response in responses))
            analysis_id = await self.analyzer.save_analysis_results(
                cluster_id, [{'analysis': response} for response in responses], selected,
                {'model': self.orchestrator.model, 'prompt': ANALYSIS_PROMPT})
            return {'status': 'success', 'cluster_id': cluster_id, 'patterns_analyzed': len(responses),
                    'successful_analyses': len(successful), 'database_analysis_id': analysis_id,
                    'sample_analysis': successful[0], 'samples': responses,
                    'errors': [response['error'] for response in responses if 'error' in response]}
        except Exception as error:
            return {'status': 'error', 'message': f'Cluster {cluster_id}: {error}', 'patterns_analyzed': len(responses)}

    async def get_analysis_summary(self, scan_name):
        try:
            rows = await self.sql_driver.execute_query(
                'SELECT la.id AS analysis_id, la.cluster_id, ic.cluster_index, ic.pattern_count AS total_patterns_in_cluster, '
                'la.representative_patterns_count, la.llm_assigned_class, la.llm_detailed_features, la.analysis_timestamp AS created_at '
                'FROM llm_analyses la JOIN identified_clusters ic ON la.cluster_id = ic.id '
                'JOIN clustering_runs cr ON ic.run_id = cr.id JOIN scans s ON cr.scan_id = s.id '
                'WHERE s.scan_name = %s ORDER BY cr.id DESC, ic.cluster_index', [scan_name])
            return {'scan_name': scan_name, 'total_clusters_analyzed': len(rows or []), 'analyses': [row.cells for row in rows or []]}
        finally:
            self.analyzer.cleanup_temp_images()
