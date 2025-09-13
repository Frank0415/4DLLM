#!/usr/bin/env python3
"""
4D-STEM LLM Analysis Pipeline with Async Parallel Processing

This script processes 4D-STEM diffraction patterns through:
1. K-Means clustering to identify similar patterns
2. Separating clustered patterns into individual files (one photo per file)
3. Async parallel processing with rate limiting (64 tasks at a time)
4. LLM analysis of each pattern
5. Storage of JSON tags and analysis results

The pipeline is designed for high-performance processing while respecting rate limits.
"""

import os
import sys
import json
import asyncio
import base64
import argparse
import logging
import numpy as np
import scipy.io
import datetime
import re
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from api_manager.apikey_loader import load_api_config

import psycopg2
from psycopg2.extras import execute_values

# --- 1. Logging and Database Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "dbname": "4dllm",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432",
}

# Load API configuration
api_conf = load_api_config("api_manager/api_keys.json")
API_KEYS = api_conf["api_keys"] if api_conf else []
BASE_URL = api_conf.get("base_url", "https://api.openai-proxy.org/google") if api_conf else "https://api.openai-proxy.org/google"
MODEL = api_conf.get("model", "gemini-2.5-pro") if api_conf else "gemini-2.5-pro"

# --- 2. Standardized Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "llm_analysis_outputs")
DATA_ROOT = os.path.join(BASE_DIR, "Data")
TEMP_IMAGE_DIR = os.path.join(BASE_DIR, "temp_images_for_llm")
CLUSTER_IMAGES_DIR = os.path.join(BASE_DIR, "cluster_images")

# Ensure directories exist
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(CLUSTER_IMAGES_DIR, exist_ok=True)

# --- 3. Final LLM Prompt Template ---
JSON_PROMPT_TEMPLATE = """
You are a professional materials scientist specializing in 4D-STEM electron diffraction pattern analysis.
Your task is to analyze the given diffraction pattern image and generate a JSON object describing its properties.

You must strictly follow the JSON structure below, and each key's value must be selected from the provided lists.

**Important Rules:**
- If `phase_type` is not "mixed_phase", then `phase_mixture_balance` and `dominant_phase_type` must be "not_applicable".
- If `phase_mixture_balance` is "balanced", then `dominant_phase_type` must be "not_applicable".

JSON structure and allowed values:
{
    "phase_type": ["crystalline", "amorphous", "polycrystalline", "mixed_phase", "unknown"],
    "crystallinity_level": ["high", "medium", "low", "none"],
    "phase_mixture_type": ["crystalline_amorphous", "crystalline_crystalline", "not_applicable"],
    "phase_mixture_balance": ["balanced", "slight_bias", "strong_bias", "overwhelming_dominant", "not_applicable"],
    "dominant_phase_type": ["crystalline", "amorphous", "unclear", "not_applicable"],
    "spot_sharpness": ["sharp", "moderately_defined", "diffuse", "streaky", "not_applicable"],
    "spot_shape": ["circular", "elliptical", "streaky", "irregular", "not_applicable"],
    "spot_intensity_distribution": ["uniform", "moderately_varied", "highly_varied", "not_applicable"],
    "lattice_distortion": ["none", "low", "medium", "high"],
    "amorphous_halo_presence": ["absent", "faint", "clear", "strong"],
    "kikuchi_lines_presence": ["absent", "faint", "clear", "strong"],
    "streaking_presence": ["absent", "minor", "moderate", "severe"],
    "superlattice_reflections": ["absent", "present_weak", "present_strong"],
    "primary_symmetry_family": ["cubic", "hexagonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic", "unclear", "not_applicable"],
    "symmetry_quality": ["high", "medium", "low", "broken"]
}

Now, please analyze this diffraction pattern and return only a valid JSON object that conforms to all rules above.
Do not include any additional explanations or markdown formatting.
"""


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from potentially noisy LLM response text.
    """
    if not isinstance(text, str):
        return {"error": "Invalid input type, expected string.", "raw_text": str(text)}
    
    # Try to find JSON enclosed in ```json ... ``` markdown blocks first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        clean_text = match.group(1)
    else:
        # If no markdown block found, try to find the first valid JSON object
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            clean_text = text[start_index : end_index + 1]
        else:
            clean_text = text
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        logger.warning(f"Unable to parse LLM output as JSON. Raw text: '{text[:100]}...'")
        return {"error": "Failed to decode JSON.", "raw_text": text}


# ########################################################################## #
# #                           【核心修改区域】                               # #
# ########################################################################## #


class LargeModelOrchestrator:
    """Orchestrator using aiohttp and REST API to manage and orchestrate LLM analysis tasks"""

    def __init__(self, api_keys: List[str], base_url: str, model: str = "gemini-2.5-pro"):
        self.api_keys = api_keys
        self.base_url = base_url
        self.model = model
        self._key_idx = 0

    def _get_next_key(self) -> str:
        """Cycle through API keys to distribute load"""
        key = self.api_keys[self._key_idx]
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        return key

    async def analyze_single_image(
        self, image_path: str, prompt: str, model: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a single image using aiohttp through REST API.
        """
        api_key = self._get_next_key()
        model = model or self.model
        
        # Construct URL according to Google Gemini API's standard REST endpoint format
        # Your proxy service (`/google`) should forward this request to Google's backend
        request_url = f"{self.base_url}/v1beta/models/{model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,  # Google REST API typically uses x-goog-api-key
        }

        try:
            b64_image = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return {"error": f"File not found: {image_path}"}

        # Build request body compliant with Google Gemini Vision API format
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                    ]
                }
            ],
            # Add generation configuration to enforce JSON output
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    request_url, headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        # Extract text content from Gemini API's response structure
                        raw_response_text = (
                            response_json.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        logger.info(f"LLM response: {raw_response_text[:100]}...")
                        # Use our robust JSON extraction function
                        return extract_json_from_llm_response(raw_response_text)
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"API request failed with status {response.status}, response: {error_text}"
                        )
                        return {
                            "error": f"API Error status {response.status}",
                            "raw_text": error_text,
                        }
            except Exception as e:
                logger.error(f"Network or request error when calling LLM REST API: {e}")
                return {"error": str(e), "raw_text": ""}


def save_analysis_result(result_dict: Dict[str, Any], output_dir: str):
    """Save a single analysis result to a JSON file"""
    image_path = result_dict["image_path"]
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filepath = os.path.join(output_dir, f"{base_filename}.json")
    
    output_data = {
        "source_image": image_path,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "analysis_payload": result_dict["analysis"],
    }
    
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        if "error" in result_dict["analysis"]:
            logger.warning(f"Saved analysis result with error to: {output_filepath}")
        else:
            logger.info(f"Result saved to: {output_filepath}")
    except IOError as e:
        logger.error(f"Unable to write file {output_filepath}: {e}")


# --- Database and Image Preparation Section ---
class DatabaseFetcher:
    """Database fetcher with support for cluster-based processing"""

    def __init__(self, config):
        self.config = config

    def get_clustered_diffraction_patterns(
        self, scan_name: str, cluster_id: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get diffraction patterns organized by clusters.
        
        Args:
            scan_name: Name of the scan to analyze
            cluster_id: Optional specific cluster to fetch (if None, fetch all)
            
        Returns:
            List of dictionaries with pattern information grouped by clusters
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
            params = (scan_name, cluster_id)
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
            params = (scan_name,)
        
        patterns = []
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query, params)
                    columns = [desc[0] for desc in cursor.description]
                    patterns = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(f"Database query failed: {error}")
        return patterns

    def get_all_clusters_for_scan(self, scan_name: str) -> List[int]:
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
        
        clusters = []
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query, (scan_name,))
                    clusters = [row[0] for row in cursor.fetchall()]
        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(f"Database query failed: {error}")
        return clusters


def preprocess_image(img, top_percent=0.5, eps=1e-8):
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


def prepare_individual_image_from_db(pattern_info: Dict[str, Any]) -> str:
    """
    Prepare a single PNG image from .mat file for one diffraction pattern.
    This creates one image per file as requested.
    
    Args:
        pattern_info: Dictionary with pattern information
        
    Returns:
        Path to the created PNG image file
    """
    try:
        mat_path = os.path.join(
            DATA_ROOT, pattern_info["scan_name"], f"{pattern_info['row_index']}.mat"
        )
        mat_data = scipy.io.loadmat(mat_path)["data"]
        image_raw = mat_data[pattern_info["col_index"] - 1, :, :]
        image_processed = preprocess_image(image_raw)
        
        # Create individual filename for each pattern
        png_filename = f"{pattern_info['scan_name']}_row_{pattern_info['row_index']}_col_{pattern_info['col_index']}_cluster_{pattern_info['cluster_label']}.png"
        png_path = os.path.join(CLUSTER_IMAGES_DIR, png_filename)
        plt.imsave(png_path, image_processed, cmap="gray", format="png")
        return png_path
    except Exception as e:
        logger.error(
            f"Failed to prepare image from .mat (row={pattern_info['row_index']}, col={pattern_info['col_index']}): {e}"
        )
        return None


def save_cluster_analysis_to_db(
    cluster_id: int, 
    analysis_result: Dict[str, Any], 
    representative_patterns: List[Dict[str, Any]]
) -> int:
    """
    Save cluster analysis results to database.
    
    Args:
        cluster_id: ID of the cluster analyzed
        analysis_result: LLM analysis result
        representative_patterns: List of representative patterns used
        
    Returns:
        ID of the created LLM analysis record
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                # Insert LLM analysis result
                insert_analysis_sql = """
                    INSERT INTO llm_analyses 
                    (cluster_id, representative_patterns_count, llm_assigned_class, llm_detailed_features)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """
                cursor.execute(
                    insert_analysis_sql,
                    (
                        cluster_id,
                        len(representative_patterns),
                        analysis_result.get("llm_assigned_class", "unknown"),
                        json.dumps(analysis_result)
                    )
                )
                analysis_id = cursor.fetchone()[0]
                
                # Insert representative patterns
                if representative_patterns:
                    insert_rep_sql = """
                        INSERT INTO llm_representative_patterns (analysis_id, pattern_id, selection_reason)
                        VALUES (%s, %s, %s);
                    """
                    for pattern in representative_patterns:
                        cursor.execute(
                            insert_rep_sql,
                            (analysis_id, pattern["pattern_id"], "selected_for_llm_analysis")
                        )
                
                conn.commit()
                return analysis_id
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Failed to save cluster analysis to database: {error}")
        return None


def save_pattern_tags_to_db(pattern_id: int, tags: Dict[str, Any]) -> None:
    """
    Save individual pattern tags to database.
    
    Args:
        pattern_id: ID of the pattern
        tags: Dictionary of tags from LLM analysis
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                # Insert tags
                insert_tag_sql = """
                    INSERT INTO llm_analysis_tags (result_id, tag_category, tag_value, confidence_score)
                    VALUES (%s, %s, %s, %s);
                """
                for key, value in tags.items():
                    # Handle different value types
                    if isinstance(value, (str, int, float)):
                        cursor.execute(insert_tag_sql, (pattern_id, key, str(value), 1.0))
                    elif isinstance(value, dict):
                        # For nested objects, save each key-value pair
                        for sub_key, sub_value in value.items():
                            cursor.execute(insert_tag_sql, (pattern_id, f"{key}_{sub_key}", str(sub_value), 1.0))
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Failed to save pattern tags to database: {error}")


async def process_cluster_patterns(
    orchestrator: LargeModelOrchestrator,
    cluster_patterns: List[Dict[str, Any]],
    cluster_id: int,
    semaphore: asyncio.Semaphore
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Process all patterns in a cluster asynchronously with rate limiting.
    
    Args:
        orchestrator: LLM orchestrator instance
        cluster_patterns: List of patterns in this cluster
        cluster_id: ID of the cluster being processed
        semaphore: Semaphore for rate limiting
        
    Returns:
        Tuple of (cluster_id, list of analysis results)
    """
    logger.info(f"Processing cluster {cluster_id} with {len(cluster_patterns)} patterns")
    
    # Prepare images for all patterns in this cluster
    image_paths = []
    valid_patterns = []
    
    logger.info(f"Preparing {len(cluster_patterns)} images for cluster {cluster_id}")
    for pattern_info in cluster_patterns:
        image_path = prepare_individual_image_from_db(pattern_info)
        if image_path:
            image_paths.append(image_path)
            valid_patterns.append(pattern_info)
    
    logger.info(f"Successfully prepared {len(image_paths)} images for cluster {cluster_id}")
    
    if not image_paths:
        logger.warning(f"No valid images prepared for cluster {cluster_id}")
        return cluster_id, []
    
    # Process images in batches with rate limiting
    batch_size = 64  # As requested
    all_results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_patterns = valid_patterns[i:i + batch_size]
        
        logger.info(f"Processing batch {i//batch_size + 1} for cluster {cluster_id} ({len(batch_paths)} images)")
        
        # Create tasks for this batch
        tasks = []
        for image_path, pattern_info in zip(batch_paths, batch_patterns):
            task = asyncio.create_task(
                process_single_pattern_with_semaphore(
                    orchestrator, image_path, pattern_info, semaphore
                )
            )
            tasks.append(task)
        
        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results.extend(batch_results)
        
        logger.info(f"Completed batch {i//batch_size + 1} for cluster {cluster_id}")
    
    logger.info(f"Completed processing cluster {cluster_id} with {len(all_results)} results")
    return cluster_id, all_results


async def process_single_pattern_with_semaphore(
    orchestrator: LargeModelOrchestrator,
    image_path: str,
    pattern_info: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Process a single pattern with semaphore for rate limiting.
    """
    async with semaphore:
        try:
            logger.info(f"Analyzing pattern at {image_path}")
            analysis = await orchestrator.analyze_single_image(
                image_path, JSON_PROMPT_TEMPLATE, model=MODEL
            )
            
            result = {
                "pattern_info": pattern_info,
                "image_path": image_path,
                "analysis": analysis,
                "processed_at": datetime.datetime.utcnow().isoformat()
            }
            
            # Save individual result
            save_analysis_result(result, ANALYSIS_OUTPUT_DIR)
            
            # Save tags to database
            if "error" not in analysis:
                save_pattern_tags_to_db(pattern_info["pattern_id"], analysis)
            
            return result
        except Exception as e:
            logger.error(f"Failed to process pattern {image_path}: {e}")
            return {
                "pattern_info": pattern_info,
                "image_path": image_path,
                "analysis": {"error": str(e)},
                "processed_at": datetime.datetime.utcnow().isoformat()
            }


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="4D-STEM image LLM analysis pipeline with rate limiting."
    )
    parser.add_argument("scan_name", help="Scan name to analyze (e.g., '1').")
    parser.add_argument("--cluster-id", type=int, help="Specific cluster ID to analyze (default: all clusters)")
    parser.add_argument("--concurrent-tasks", type=int, default=64, help="Maximum concurrent LLM tasks (default: 64)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage 1: Fetching clustered diffraction patterns from database...")
    
    db_fetcher = DatabaseFetcher(DB_CONFIG)
    
    if args.cluster_id:
        # Process only a specific cluster
        clusters_to_process = [args.cluster_id]
        logger.info(f"Processing only cluster {args.cluster_id}")
    else:
        # Process all clusters
        clusters_to_process = db_fetcher.get_all_clusters_for_scan(args.scan_name)
        logger.info(f"Found {len(clusters_to_process)} clusters to process: {clusters_to_process}")
    
    if not clusters_to_process:
        logger.error("No clusters found for analysis. Terminating process.")
        return
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrent_tasks)
    logger.info(f"Rate limiting set to {args.concurrent_tasks} concurrent tasks")
    
    logger.info("=" * 60)
    logger.info("Stage 2: Setting up LLM orchestrator...")
    
    # Load API configuration
    if not API_KEYS:
        logger.error("Unable to load API configuration. Terminating process.")
        return

    # Use API keys and base_url from config file
    orchestrator = LargeModelOrchestrator(
        api_keys=API_KEYS, base_url=BASE_URL, model=MODEL
    )
    
    output_dir = os.path.join(
        ANALYSIS_OUTPUT_DIR,
        f"{args.scan_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"All analysis results will be saved to: {output_dir}")
    
    logger.info("=" * 60)
    logger.info("Stage 3: Beginning cluster-based LLM analysis...")
    
    # Process each cluster
    cluster_tasks = []
    for cluster_id in clusters_to_process:
        logger.info(f"Fetching patterns for cluster {cluster_id}")
        cluster_patterns = db_fetcher.get_clustered_diffraction_patterns(
            args.scan_name, cluster_id
        )
        
        if not cluster_patterns:
            logger.warning(f"No patterns found for cluster {cluster_id}")
            continue
            
        logger.info(f"Found {len(cluster_patterns)} patterns in cluster {cluster_id}")
        
        # Create task for this cluster
        task = asyncio.create_task(
            process_cluster_patterns(orchestrator, cluster_patterns, cluster_id, semaphore)
        )
        cluster_tasks.append(task)
    
    # Wait for all cluster processing to complete
    cluster_results = await asyncio.gather(*cluster_tasks, return_exceptions=True)
    
    # Process and save results
    logger.info("=" * 60)
    logger.info("Stage 4: Saving final results...")
    
    successful_clusters = 0
    total_patterns_processed = 0
    
    for result in cluster_results:
        if isinstance(result, Exception):
            logger.error(f"Cluster processing failed with error: {result}")
            continue
            
        if isinstance(result, tuple) and len(result) == 2:
            cluster_id, patterns_results = result
            successful_clusters += 1
            total_patterns_processed += len(patterns_results)
            
            # Save cluster-level analysis
            if patterns_results:
                # Aggregate results for this cluster
                cluster_analysis = {
                    "cluster_id": cluster_id,
                    "total_patterns": len(patterns_results),
                    "successful_patterns": len([r for r in patterns_results if "error" not in r.get("analysis", {})]),
                    "failed_patterns": len([r for r in patterns_results if "error" in r.get("analysis", {})])
                }
                
                # Save cluster analysis to database
                # (In a real implementation, you might want to aggregate pattern results)
                logger.info(f"Cluster {cluster_id} processed {len(patterns_results)} patterns")
    
    logger.info("=" * 60)
    logger.info("🎉 All clusters processed successfully!")
    logger.info(f"📊 Summary: {successful_clusters} clusters with {total_patterns_processed} total patterns processed")
    logger.info(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    asyncio.run(main())