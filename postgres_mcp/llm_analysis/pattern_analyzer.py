"""
Pattern Analyzer for processing 4D-STEM diffraction patterns.
"""

import json
import logging
import numpy as np
import os
import scipy.io
from pathlib import Path
from typing import Dict, Any, List, Union
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from ..sql import SqlDriver, SafeSqlDriver

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Analyzer for processing and preparing diffraction patterns for LLM analysis."""

    def __init__(
        self, sql_driver: Union[SqlDriver, SafeSqlDriver], data_root: str = None
    ):
        """
        Initialize pattern analyzer.

        Args:
            sql_driver: Database driver instance
            data_root: Root directory for data files
        """
        self.sql_driver = sql_driver
        self.data_root = data_root or os.path.join(
            os.path.dirname(__file__), "..", "..", "Data"
        )

        # Create directories for image processing in current folder
        self.current_dir = Path(os.getcwd()) / "cluster_images"
        self.cluster_dir = self.current_dir

        self.current_dir.mkdir(exist_ok=True)
        self.cluster_dir.mkdir(exist_ok=True)

    async def get_clustered_patterns(
        self, scan_name: str, cluster_id: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get diffraction patterns organized by clusters.

        Args:
            scan_name: Name of the scan to analyze
            cluster_id: Optional specific cluster to fetch

        Returns:
            List of pattern information dictionaries
        """
        if cluster_id is not None:
            sql_query = """
                SELECT dp.id AS pattern_id, rmf.row_index, dp.col_index, 
                       dp.cluster_label, s.scan_name
                FROM diffraction_patterns dp
                JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
                JOIN scans s ON rmf.scan_id = s.id
                WHERE s.scan_name = %s AND dp.cluster_label = %s
                ORDER BY rmf.row_index, dp.col_index;
            """
            params = [scan_name, cluster_id]
        else:
            sql_query = """
                SELECT dp.id AS pattern_id, rmf.row_index, dp.col_index, 
                       dp.cluster_label, s.scan_name
                FROM diffraction_patterns dp
                JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
                JOIN scans s ON rmf.scan_id = s.id
                WHERE s.scan_name = %s AND dp.cluster_label IS NOT NULL
                ORDER BY dp.cluster_label, rmf.row_index, dp.col_index;
            """
            params = [scan_name]

        result = await self.sql_driver.execute_query(sql_query, params)

        if not result:
            return []

        patterns = []
        for row in result:
            patterns.append(
                {
                    "pattern_id": row.cells["pattern_id"],
                    "row_index": row.cells["row_index"],
                    "col_index": row.cells["col_index"],
                    "cluster_label": row.cells["cluster_label"],
                    "scan_name": row.cells["scan_name"],
                }
            )

        return patterns

    async def get_all_clusters_for_scan(self, scan_name: str) -> List[int]:
        """
        Get all unique cluster IDs for a scan.

        Args:
            scan_name: Name of the scan

        Returns:
            List of unique cluster IDs
        """
        sql_query = """
            SELECT DISTINCT dp.cluster_label
            FROM diffraction_patterns dp
            JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
            JOIN scans s ON rmf.scan_id = s.id
            WHERE s.scan_name = %s AND dp.cluster_label IS NOT NULL
            ORDER BY dp.cluster_label;
        """

        result = await self.sql_driver.execute_query(sql_query, [scan_name])

        if not result:
            return []

        return [row.cells["cluster_label"] for row in result]

    def prepare_pattern_image(self, pattern_info: Dict[str, Any]) -> str:
        """
        Prepare a single PNG image from .mat file for one diffraction pattern.

        Args:
            pattern_info: Dictionary with pattern information

        Returns:
            Path to the created PNG image file
        """
        try:
            mat_path = os.path.join(
                self.data_root,
                pattern_info["scan_name"],
                f"{pattern_info['row_index']}.mat",
            )

            if not os.path.exists(mat_path):
                logger.error(f"Mat file not found: {mat_path}")
                return None

            mat_data = scipy.io.loadmat(mat_path)["data"]
            image_raw = mat_data[pattern_info["col_index"] - 1, :, :]
            image_processed = self._preprocess_image(image_raw)

            # Create individual filename for each pattern
            png_filename = (
                f"{pattern_info['scan_name']}_row_{pattern_info['row_index']}_"
                f"col_{pattern_info['col_index']}_cluster_{pattern_info['cluster_label']}.png"
            )
            png_path = self.cluster_dir / png_filename

            plt.imsave(str(png_path), image_processed, cmap="gray", format="png")
            logger.info(f"Created pattern image: {png_path}")
            return str(png_path)

        except Exception as e:
            logger.error(
                f"Failed to prepare image from .mat "
                f"(row={pattern_info['row_index']}, col={pattern_info['col_index']}): {e}"
            )
            return None

    def _preprocess_image(self, img, top_percent=0.5, eps=1e-8):
        """
        Preprocess raw diffraction image for visualization and saving.
        - Clip the brightest top_percent pixels
        - Normalize to [0, 1] range
        """
        if img is None:
            return np.zeros((224, 224), dtype=np.float32)

        img = img.astype(np.float32)

        if top_percent > 0 and np.max(img) > 0:
            threshold = np.percentile(img, 100 - top_percent)
            img = np.clip(img, None, threshold)

        min_val, max_val = img.min(), img.max()
        if max_val - min_val > eps:
            img_normalized = (img - min_val) / (max_val - min_val)
        else:
            img_normalized = np.zeros_like(img, dtype=np.float32)

        return img_normalized

    async def save_analysis_results(
        self,
        cluster_id: int,
        analysis_results: List[Dict[str, Any]],
        representative_patterns: List[Dict[str, Any]] = None,
    ) -> int:
        """
        Save cluster analysis results to database.

        Args:
            cluster_id: ID of the cluster analyzed
            analysis_results: List of LLM analysis results
            representative_patterns: List of representative patterns used

        Returns:
            ID of the created LLM analysis record
        """
        try:
            # Aggregate analysis results
            successful_analyses = [
                r for r in analysis_results if "error" not in r.get("analysis", {})
            ]

            if not successful_analyses:
                logger.warning(f"No successful analyses for cluster {cluster_id}")
                return None

            # Take the first successful analysis as representative
            # In a more sophisticated implementation, you might aggregate multiple results
            representative_analysis = successful_analyses[0]["analysis"]

            # Insert LLM analysis result
            insert_analysis_sql = """
                INSERT INTO llm_analyses 
                (cluster_id, representative_patterns_count, llm_assigned_class, llm_detailed_features)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """

            result = await self.sql_driver.execute_query(
                insert_analysis_sql,
                [
                    cluster_id,
                    len(representative_patterns or []),
                    representative_analysis.get("phase_type", "unknown"),
                    json.dumps(representative_analysis),
                ],
            )

            if not result:
                raise RuntimeError("Failed to insert LLM analysis")

            analysis_id = result[0].cells["id"]

            # Insert representative patterns if provided
            if representative_patterns:
                insert_rep_sql = """
                    INSERT INTO llm_representative_patterns (analysis_id, pattern_id, selection_reason)
                    VALUES (%s, %s, %s);
                """
                for pattern in representative_patterns:
                    await self.sql_driver.execute_query(
                        insert_rep_sql,
                        [
                            analysis_id,
                            pattern["pattern_id"],
                            "selected_for_llm_analysis",
                        ],
                    )

            logger.info(
                f"Saved cluster analysis for cluster {cluster_id} with ID: {analysis_id}"
            )
            return analysis_id

        except Exception as e:
            logger.error(f"Failed to save cluster analysis to database: {e}")
            return None

    async def save_pattern_tags(self, pattern_id: int, tags: Dict[str, Any]) -> None:
        """
        Save individual pattern tags to database.

        Args:
            pattern_id: ID of the pattern
            tags: Dictionary of tags from LLM analysis
        """
        try:
            # Insert tags
            insert_tag_sql = """
                INSERT INTO llm_analysis_tags (result_id, tag_category, tag_value, confidence_score)
                VALUES (%s, %s, %s, %s);
            """

            for key, value in tags.items():
                if key == "error":
                    continue  # Skip error fields

                # Handle different value types
                if isinstance(value, (str, int, float)):
                    await self.sql_driver.execute_query(
                        insert_tag_sql, [pattern_id, key, str(value), 1.0]
                    )
                elif isinstance(value, dict):
                    # For nested objects, save each key-value pair
                    for sub_key, sub_value in value.items():
                        await self.sql_driver.execute_query(
                            insert_tag_sql,
                            [pattern_id, f"{key}_{sub_key}", str(sub_value), 1.0],
                        )

        except Exception as e:
            logger.error(f"Failed to save pattern tags to database: {e}")

    def cleanup_temp_images(self):
        """Clean up temporary image files."""
        try:
            for cluster_file in self.cluster_dir.glob("*.png"):
                cluster_file.unlink()
            logger.info("Cleaned up cluster image files")
        except Exception as e:
            logger.warning(f"Failed to clean up cluster files: {e}")
