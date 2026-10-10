import json

from .analysis_pipeline import AnalysisPipeline
from .pattern_analyzer import CATEGORIES

CONSENSUS_PROMPT_TEMPLATE = '''Summarize the shared observed diffraction features of these patterns from one
cluster. Separate common evidence from uncertain hypotheses. Return a JSON object
with consensus_description and dominant_classification_code (0 vacuum,
1 crystalline, 2 amorphous, 3 mixed). Use the observed classifications below:
'''


class ConsensusAnalyzer:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    async def generate_consensus(self, cluster_id, pattern_descriptions, model=None):
        if not pattern_descriptions:
            raise ValueError('No successful pattern descriptions for consensus')
        result = await self.orchestrator.analyze_text_prompt(
            CONSENSUS_PROMPT_TEMPLATE + json.dumps(pattern_descriptions), model)
        if 'error' in result:
            raise ValueError(result['error'])
        code = result.get('dominant_classification_code')
        if type(code) is not int or code not in CATEGORIES:
            raise ValueError('LLM consensus returned an invalid classification code')
        return {'cluster_id': cluster_id, 'consensus_description': result.get('consensus_description', ''),
                'dominant_classification_code': code, 'pattern_count': len(pattern_descriptions)}

    async def analyze_cluster_patterns(self, sql_driver, scan_id, clustering_run_id, cluster_id, cluster_index, model=None):
        rows = await sql_driver.execute_query('SELECT scan_name FROM scans WHERE id = %s', [scan_id])
        if not rows:
            raise ValueError('Scan not found')
        pipeline = AnalysisPipeline(sql_driver, self.orchestrator.api_keys, self.orchestrator.base_url,
                                    model or self.orchestrator.model)
        try:
            analyzed = await pipeline._analyze_single_cluster(rows[0].cells['scan_name'], cluster_id, 16, 3)
            if analyzed['status'] != 'success':
                raise ValueError(analyzed['message'])
            samples = [sample for sample in analyzed['samples'] if 'error' not in sample]
            result = await self.generate_consensus(cluster_id, samples, model)
            result['cluster_index'] = cluster_index
            return result
        finally:
            pipeline.analyzer.cleanup_temp_images()

    async def store_consensus_results(self, sql_driver, scan_id, clustering_run_id, consensus_results):
        async with sql_driver.transaction() as tx:
            for result in consensus_results:
                details = {key: result[key] for key in ('consensus_description', 'dominant_classification_code', 'pattern_count')}
                await tx.execute_query(
                    "UPDATE llm_analyses la SET llm_assigned_class = %s, "
                    "llm_detailed_features = COALESCE(la.llm_detailed_features, '{}'::jsonb) || %s::jsonb "
                    "FROM identified_clusters ic JOIN clustering_runs cr ON ic.run_id = cr.id "
                    "WHERE la.cluster_id = ic.id AND ic.id = %s AND ic.run_id = %s AND cr.scan_id = %s",
                    [CATEGORIES[result['dominant_classification_code']], json.dumps(details), result['cluster_id'], clustering_run_id, scan_id])
