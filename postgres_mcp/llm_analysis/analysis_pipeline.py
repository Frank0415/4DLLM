"""
Complete Analysis Pipeline for LLM-based 4D-STEM pattern analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .llm_orchestrator import LLMOrchestrator
from .pattern_analyzer import PatternAnalyzer
from ..sql import SqlDriver, SafeSqlDriver

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Complete pipeline for LLM-based cluster analysis of 4D-STEM patterns."""

    def __init__(
        self,
        sql_driver: Union[SqlDriver, SafeSqlDriver],
        api_keys: List[str],
        base_url: str = "https://api.example.com",
        model: str = "default-model",
        data_root: str = None,
    ):
        """
        Initialize the analysis pipeline.

        Args:
            sql_driver: Database driver instance
            api_keys: List of API keys for LLM analysis
            base_url: Base URL for the LLM API
            model: Model name to use
            data_root: Root directory for data files
        """
        self.sql_driver = sql_driver
        self.orchestrator = LLMOrchestrator(api_keys, base_url, model)
        self.analyzer = PatternAnalyzer(sql_driver, data_root)

    async def analyze_scan_clusters(
        self,
        scan_name: str,
        cluster_id: Optional[int] = None,
        max_patterns_per_cluster: int = 5,
        batch_size: int = 3,
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a scan's clusters.

        Args:
            scan_name: Name of the scan to analyze
            cluster_id: Optional specific cluster ID to analyze (None = all clusters)
            max_patterns_per_cluster: Maximum patterns to analyze per cluster
            batch_size: Number of patterns to process in parallel

        Returns:
            Dictionary with analysis results and statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting LLM analysis pipeline for scan: {scan_name}")

        try:
            # Step 1: Get clusters to analyze
            if cluster_id is not None:
                clusters_to_analyze = [cluster_id]
            else:
                clusters_to_analyze = await self.analyzer.get_all_clusters_for_scan(
                    scan_name
                )

            if not clusters_to_analyze:
                return {
                    "status": "error",
                    "message": f"No clusters found for scan: {scan_name}",
                    "clusters_analyzed": 0,
                    "total_patterns_analyzed": 0,
                }

            logger.info(f"Found {len(clusters_to_analyze)} clusters to analyze")

            # Step 2: Analyze each cluster
            pipeline_results = {
                "status": "success",
                "scan_name": scan_name,
                "start_time": start_time.isoformat(),
                "clusters_analyzed": 0,
                "total_patterns_analyzed": 0,
                "cluster_results": {},
                "errors": [],
            }

            for cluster_id in clusters_to_analyze:
                try:
                    cluster_result = await self._analyze_single_cluster(
                        scan_name, cluster_id, max_patterns_per_cluster, batch_size
                    )

                    pipeline_results["cluster_results"][str(cluster_id)] = (
                        cluster_result
                    )
                    pipeline_results["clusters_analyzed"] += 1
                    pipeline_results["total_patterns_analyzed"] += cluster_result.get(
                        "patterns_analyzed", 0
                    )

                    logger.info(
                        f"Completed cluster {cluster_id}: {cluster_result.get('patterns_analyzed', 0)} patterns"
                    )

                except Exception as e:
                    error_msg = f"Failed to analyze cluster {cluster_id}: {str(e)}"
                    logger.error(error_msg)
                    pipeline_results["errors"].append(error_msg)

            # Step 3: Cleanup and final stats
            self.analyzer.cleanup_temp_images()

            end_time = datetime.now()
            pipeline_results["end_time"] = end_time.isoformat()
            pipeline_results["duration_seconds"] = (
                end_time - start_time
            ).total_seconds()

            logger.info(
                f"Pipeline completed: {pipeline_results['clusters_analyzed']} clusters, "
                f"{pipeline_results['total_patterns_analyzed']} patterns in "
                f"{pipeline_results['duration_seconds']:.1f}s"
            )

            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "clusters_analyzed": 0,
                "total_patterns_analyzed": 0,
            }

    async def _analyze_single_cluster(
        self,
        scan_name: str,
        cluster_id: int,
        max_patterns_per_cluster: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Analyze patterns from a single cluster.

        Args:
            scan_name: Name of the scan
            cluster_id: ID of the cluster to analyze
            max_patterns_per_cluster: Maximum patterns to include
            batch_size: Batch size for parallel processing

        Returns:
            Dictionary with cluster analysis results
        """
        logger.info(f"Analyzing cluster {cluster_id} from scan {scan_name}")

        # Step 1: Get patterns for this cluster
        patterns = await self.analyzer.get_clustered_patterns(scan_name, cluster_id)

        if not patterns:
            return {
                "status": "error",
                "message": f"No patterns found for cluster {cluster_id}",
                "patterns_analyzed": 0,
            }

        # Step 2: Limit patterns if needed and select representative ones
        if len(patterns) > max_patterns_per_cluster:
            # Select patterns with good spatial distribution
            selected_patterns = self._select_representative_patterns(
                patterns, max_patterns_per_cluster
            )
        else:
            selected_patterns = patterns

        logger.info(
            f"Selected {len(selected_patterns)} patterns from cluster {cluster_id}"
        )

        # Step 3: Prepare images for LLM analysis
        image_paths = []
        prepared_patterns = []

        for pattern in selected_patterns:
            image_path = self.analyzer.prepare_pattern_image(pattern)
            if image_path:
                image_paths.append(image_path)
                prepared_patterns.append(pattern)

        if not image_paths:
            return {
                "status": "error",
                "message": f"Failed to prepare any images for cluster {cluster_id}",
                "patterns_analyzed": 0,
            }

        # Step 4: Run LLM analysis in batches
        try:
            llm_results = await self.orchestrator.analyze_batch(
                image_paths,
                batch_size=batch_size,
                analysis_prompt=self._get_analysis_prompt(),
            )

            # Step 5: Save results to database
            analysis_id = await self.analyzer.save_analysis_results(
                cluster_id, llm_results, prepared_patterns
            )

            # Step 6: Save individual pattern tags
            for i, (pattern, result) in enumerate(zip(prepared_patterns, llm_results)):
                if "error" not in result.get("analysis", {}):
                    await self.analyzer.save_pattern_tags(
                        pattern["pattern_id"], result["analysis"]
                    )

            # Step 7: Compile results
            successful_results = [
                r for r in llm_results if "error" not in r.get("analysis", {})
            ]

            return {
                "status": "success",
                "cluster_id": cluster_id,
                "patterns_analyzed": len(llm_results),
                "successful_analyses": len(successful_results),
                "database_analysis_id": analysis_id,
                "sample_analysis": successful_results[0]["analysis"]
                if successful_results
                else None,
                "errors": [
                    r["analysis"].get("error")
                    for r in llm_results
                    if "error" in r.get("analysis", {})
                ],
            }

        except Exception as e:
            logger.error(f"LLM analysis failed for cluster {cluster_id}: {str(e)}")
            return {"status": "error", "message": str(e), "patterns_analyzed": 0}

    def _select_representative_patterns(
        self, patterns: List[Dict[str, Any]], max_count: int
    ) -> List[Dict[str, Any]]:
        """
        Select representative patterns from a cluster for analysis.
        Uses spatial distribution to pick diverse patterns.

        Args:
            patterns: List of all patterns in cluster
            max_count: Maximum number to select

        Returns:
            List of selected representative patterns
        """
        if len(patterns) <= max_count:
            return patterns

        # Sort by row and column for spatial distribution
        patterns_sorted = sorted(
            patterns, key=lambda p: (p["row_index"], p["col_index"])
        )

        # Select evenly spaced patterns
        step = len(patterns_sorted) // max_count
        selected = []

        for i in range(0, len(patterns_sorted), step):
            if len(selected) < max_count:
                selected.append(patterns_sorted[i])

        # If we're still short, add some from the end
        while len(selected) < max_count and len(selected) < len(patterns_sorted):
            for pattern in reversed(patterns_sorted):
                if pattern not in selected:
                    selected.append(pattern)
                    break

        logger.info(
            f"Selected {len(selected)} representative patterns from {len(patterns)} total patterns"
        )
        return selected

    def _get_analysis_prompt(self) -> str:
        """Get the analysis prompt for LLM."""
        return """
Analyze this 4D-STEM diffraction pattern image. Please provide:

1. Phase Type: Identify the likely crystalline phase or material type
2. Structural Features: Describe key diffraction features (spots, rings, streaks, etc.)
3. Symmetry: Comment on apparent symmetry in the pattern
4. Quality Assessment: Rate the pattern quality (clear/noisy/damaged)
5. Notable Features: Any special characteristics or anomalies

Please respond with a JSON object containing:
{
    "phase_type": "description of likely phase/material",
    "structural_features": "description of diffraction features",
    "symmetry": "symmetry observations", 
    "quality": "quality assessment",
    "notable_features": "any special characteristics",
    "confidence": "high/medium/low confidence in analysis"
}
"""

    async def get_analysis_summary(self, scan_name: str) -> Dict[str, Any]:
        """
        Get a summary of all LLM analyses for a scan.

        Args:
            scan_name: Name of the scan

        Returns:
            Dictionary with analysis summary
        """
        try:
            # Get all LLM analyses for this scan
            summary_sql = """
                SELECT la.id, la.cluster_id, la.representative_patterns_count,
                       la.llm_assigned_class, la.llm_detailed_features,
                       la.analysis_timestamp, COUNT(dp.id) as total_patterns_in_cluster
                FROM llm_analyses la
                JOIN diffraction_patterns dp ON la.cluster_id = dp.cluster_label
                JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
                JOIN scans s ON rmf.scan_id = s.id
                WHERE s.scan_name = %s
                GROUP BY la.id, la.cluster_id, la.representative_patterns_count,
                         la.llm_assigned_class, la.llm_detailed_features, la.analysis_timestamp
                ORDER BY la.cluster_id;
            """

            result = await self.sql_driver.execute_query(summary_sql, [scan_name])

            if not result:
                return {
                    "scan_name": scan_name,
                    "total_clusters_analyzed": 0,
                    "analyses": [],
                }

            analyses = []
            for row in result:
                analyses.append(
                    {
                        "analysis_id": row.cells["id"],
                        "cluster_id": row.cells["cluster_id"],
                        "representative_patterns_count": row.cells[
                            "representative_patterns_count"
                        ],
                        "llm_assigned_class": row.cells["llm_assigned_class"],
                        "total_patterns_in_cluster": row.cells[
                            "total_patterns_in_cluster"
                        ],
                        "created_at": row.cells["analysis_timestamp"],
                    }
                )

            return {
                "scan_name": scan_name,
                "total_clusters_analyzed": len(analyses),
                "analyses": analyses,
            }

        except Exception as e:
            logger.error(f"Failed to get analysis summary: {str(e)}")
            return {"scan_name": scan_name, "error": str(e)}
