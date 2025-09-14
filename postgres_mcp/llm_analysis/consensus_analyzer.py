"""
LLM Consensus Analyzer for generating consensus descriptions for k-means clusters.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the path so we can import postgres_mcp modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Set up unbuffered logging to /tmp/llm_logging with rotation
log_file_path = "/tmp/llm_logging"
# Check if log file exists and is larger than 10MB, if so, rotate it
if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 10 * 1024 * 1024:
    # Rotate the log file
    if os.path.exists(f"{log_file_path}.1"):
        os.remove(f"{log_file_path}.1")
    os.rename(log_file_path, f"{log_file_path}.1")

file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(formatter)
# Force unbuffered output
file_handler.stream.reconfigure(line_buffering=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False  # Don't propagate to root logger

from postgres_mcp.llm_analysis.llm_orchestrator import LLMOrchestrator
from postgres_mcp.sql import SafeSqlDriver

# Consensus prompt template for generating cluster descriptions
CONSENSUS_PROMPT_TEMPLATE = """
**Role**
Act as an expert materials scientist specializing in the advanced analysis of 4D-STEM and electron diffraction data. Your expertise lies in interpreting complex diffraction patterns, including Convergent Beam Electron Diffraction (CBED), to determine local crystal structure, orientation, and phase.

**Objective**
Your task is to analyze a set of 16 individual descriptions of CBED patterns obtained from various locations within the same cluster/category. You must identify the recurring, common features across all 16 descriptions and synthesize these findings into a single, cohesive summary paragraph that represents the defining characteristics of this cluster.

**Input Data**
I will provide you with 16 JSON objects. Each object contains a classification_code and a description of a single CBED pattern. All of these patterns belong to the same cluster/category.

**Key Instructions**
1. Focus on Commonalities: Your summary must only describe features that are consistently present across the entire set of 16 descriptions. Do not include characteristics that appear in only a few patterns.

2. Maintain a Professional Tone: Use precise, objective, and technical language appropriate for a materials science publication.

3. Handle Variability: The full dataset of 16 patterns may describe a variety of states, including crystalline, amorphous, mixed-phase, or even empty/vacuum regions. Your final summary must accurately reflect the common ground of the entire dataset.

4. Structure your response as a JSON object with two keys:
   - "consensus_description": A single, objective paragraph that describes the defining characteristics of this cluster
   - "dominant_classification_code": The most frequently occurring classification code among the 16 samples

Here are the 16 JSON objects:
"""


class ConsensusAnalyzer:
    """Analyzer for generating consensus descriptions for k-means clusters using LLM"""

    def __init__(self, orchestrator: LLMOrchestrator):
        """
        Initialize the consensus analyzer.

        Args:
            orchestrator: LLM orchestrator instance for making API calls
        """
        self.orchestrator = orchestrator
        logger.info("Initialized ConsensusAnalyzer")

    def _preprocess_image(self, img, top_percent=0.5, eps=1e-8, crop_size=224):
        """
        Preprocess image from mat file - same as server.py function.
        从中心裁剪到 crop_size × crop_size，
        然后去掉最亮 top_percent 百分比的像素，
        再归一到 [0,1]
        """
        import numpy as np

        img = img.astype(np.float32)

        # 中心裁剪
        H, W = img.shape
        ch = crop_size // 2
        center_row = H // 2
        center_col = W // 2
        img_cropped = img[
            max(center_row - ch, 0) : min(center_row + ch, H),
            max(center_col - ch, 0) : min(center_col + ch, W),
        ]

        # 如果裁出来不足 crop_size，再补零（padding）
        if img_cropped.shape[0] != crop_size or img_cropped.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size), dtype=np.float32)
            h, w = img_cropped.shape
            padded[:h, :w] = img_cropped
            img_cropped = padded

        # 再做 top_percent 剪切
        if top_percent > 0:
            threshold = np.percentile(img_cropped, 100 - top_percent)
            img_cropped = np.clip(img_cropped, None, threshold)

        mn, mx = img_cropped.min(), img_cropped.max()
        img_cropped = (img_cropped - mn) / (mx - mn + eps)

        return img_cropped

    async def generate_cluster_consensus(
        self,
        cluster_id: int,
        pattern_descriptions: List[Dict[str, Any]],
        model: str = None,
    ) -> Dict[str, Any]:
        """
        Generate a consensus description for a cluster based on individual pattern descriptions.

        Args:
            cluster_id: The ID of the cluster
            pattern_descriptions: List of dictionaries containing 'classification_code' and 'description'
            model: Optional model override

        Returns:
            Dictionary containing consensus description and dominant classification code
        """
        logger.info(
            f"Generating consensus for cluster_id={cluster_id} with {len(pattern_descriptions)} patterns"
        )

        # Prepare the prompt with all pattern descriptions
        prompt = CONSENSUS_PROMPT_TEMPLATE + json.dumps(pattern_descriptions, indent=2)
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Log the first pattern description for debugging
        if pattern_descriptions:
            logger.debug(f"First pattern sample: {pattern_descriptions[0]}")

        # Call the LLM to generate consensus
        logger.info(f"Calling LLM for consensus analysis of cluster {cluster_id}")
        result = await self.orchestrator.analyze_text_prompt(
            prompt, model or self.orchestrator.model
        )
        logger.info(f"Received LLM response for cluster {cluster_id}")

        # Log the raw result for debugging
        logger.debug(f"Raw LLM result: {result}")

        # Extract consensus information
        consensus_description = result.get("consensus_description", "")
        dominant_code = result.get("dominant_classification_code", -1)

        logger.info(
            f"Extracted consensus for cluster {cluster_id}: description length={len(consensus_description)}, dominant_code={dominant_code}"
        )

        return {
            "cluster_id": cluster_id,
            "consensus_description": consensus_description,
            "dominant_classification_code": dominant_code,
            "pattern_count": len(pattern_descriptions),
        }

    async def analyze_cluster_patterns(
        self,
        sql_driver: SafeSqlDriver,
        scan_id: int,
        clustering_run_id: int,
        cluster_id: int,
        cluster_index: int,
        model: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze patterns in a cluster and generate consensus description.

        Args:
            sql_driver: Database driver for querying pattern data
            scan_id: ID of the scan
            clustering_run_id: ID of the clustering run
            cluster_id: Database ID of the cluster
            cluster_index: Index of the cluster (0-based)
            model: Optional model override

        Returns:
            Dictionary containing analysis results
        """
        logger.info(
            f"Analyzing cluster patterns for scan_id={scan_id}, clustering_run_id={clustering_run_id}, cluster_id={cluster_id}, cluster_index={cluster_index}"
        )

        # Get 16 representative patterns from this cluster
        query = """
            SELECT dp.id, dp.source_mat_id, dp.col_index, dp.cluster_label
            FROM diffraction_patterns dp
            JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
            WHERE dp.clustering_run_id = %s AND dp.cluster_label = %s
            ORDER BY RANDOM()
            LIMIT 16
        """

        logger.info(f"Executing query to get patterns for cluster {cluster_index}")
        rows = await sql_driver.execute_query(query, [clustering_run_id, cluster_index])
        logger.info(
            f"Retrieved {len(rows) if rows else 0} patterns for cluster {cluster_index}"
        )

        if not rows:
            logger.warning(
                f"No patterns found for cluster {cluster_index} in clustering run {clustering_run_id}"
            )
            return {
                "cluster_id": cluster_id,
                "cluster_index": cluster_index,
                "consensus_description": "No patterns found for this cluster",
                "dominant_classification_code": -1,
                "pattern_count": 0,
            }

            # Try to get actual LLM analysis results for these patterns
        pattern_descriptions = []
        temp_images_dir = "/tmp/consensus_analysis_images"
        os.makedirs(temp_images_dir, exist_ok=True)

        JSON_PROMPT_TEMPLATE = """
You are an expert materials scientist specializing in the analysis of 4D-STEM and electron diffraction data. Your primary task is to meticulously analyze the provided 2D image, which is a single convergent beam electron diffraction (CBED) pattern, and classify the material's state at the probed location.

**Your classification must result in one of the following four categories, which you will map to a numeric code:**
*   `0: Vacuum` (The electron beam did not interact with the sample.)
*   `1: Crystalline` (The beam interacted with a region of long-range atomic order.)
*   `2: Amorphous` (The beam interacted with a region of short-range atomic order.)
*   `3: Mixed-State` (The beam interacted with a region containing both crystalline and amorphous phases.)

**INTERNAL ANALYSIS FRAMEWORK (Use this for your reasoning but do not include it in the output):**

Your internal analysis must follow two sequential steps: Observation followed by Logical Deduction.

**Part 1: Feature Observation Checklist**
First, systematically evaluate the image against the six key features in the table below. This provides the raw observational evidence for your analysis.

| Feature / Criterion | **Vacuum** | **Crystalline** | **Amorphous** | **Mixed-State** |
| :--- | :--- | :--- | :--- | :--- |
| **1. Central Transmitted Beam** | Present and is the **sole feature**. | Present, sharp, and typically the brightest point. | Present, but may blend with the inner diffuse signal. | Present and clearly visible. |
| **2. Sharp Bragg Peaks** | **Completely Absent.** | **Present.** These are discrete, **sharp, point-like** spots. This is the defining feature of a crystalline component. | **Completely Absent.** The pattern may have bright *speckles* or *intensity maxima*, but these are **not** sharp, point-like Bragg peaks. | **Present.** True, sharp, point-like Bragg peaks are clearly visible. |
| **3. Geometric Arrangement** | Not Applicable (N/A). | Peaks are arranged in a pattern with **clear crystallographic point group symmetry** (e.g., hexagonal, square). | Features are arranged with **only radial symmetry** (i.e., in a ring). Any speckles/maxima show no crystallographic symmetry. | Crystalline peaks show crystallographic symmetry, overlaid on a radially symmetric amorphous signal. |
| **4. Broad Diffuse Halo** | **Completely Absent.** | **Absent.** Faint Thermal Diffuse Scattering (TDS) may exist but **no distinct, coherent, radially symmetric ring.** | **Present.** This is the defining feature. It is a distinct ring of intensity with clear **radial symmetry**. This halo may be **(a) smooth and continuous** OR **(b) a grainy ring of speckles/intensity fluctuations** (the typical appearance when probed by a small, coherent beam). | **Present.** A distinct, radially symmetric halo (either smooth or speckled) is clearly visible in the background *between* the sharp Bragg peaks. |
| **5. Feature Superposition** | Not Applicable (N/A). | Not Applicable (N/A). | Not Applicable (N/A). | **Observed.** Sharp Bragg peaks (Feature 2) are clearly overlaid on a **distinct, radially symmetric halo** (Feature 4). |
| **6. Background Signal** | Uniformly dark except for the central beam. | **Largely dark and clean** between discrete spots. (Does not apply to complex zone axes with many overlapping disks/lines). | The "background" *is* the diffuse ring or speckle pattern itself. | **Elevated and structured.** The signal *between* the sharp Bragg peaks forms a **coherent, radially symmetric halo/ring (smooth or speckled)**. |

---
**Part 2: Decision-Making Logic**
Next, apply the following hierarchical rules to your observations from the Feature Checklist to reach a final classification.

*   **Rule 1 (Vacuum):** The classification is **Vacuum** if Feature 1 is `Observed` AND Features 2 and 4 are `Not Observed`.
*   **Rule 2 (Crystalline):** The classification is **Crystalline** if sharp Bragg peaks (Feature 2) are `Observed` AND these peaks exhibit crystallographic point group symmetry (Feature 3), AND a true amorphous halo (Feature 4) is `Not Observed`.
*   **Rule 3 (Amorphous):** The classification is **Amorphous** if a Broad Diffuse Halo (Feature 4) is `Observed`, AND true, sharp Bragg peaks (Feature 2) are definitively `Not Observed`. The observed pattern consists only of the central beam and this diffuse/speckled ring, which lacks crystallographic symmetry (per Feature 3).
*   **Rule 4 (Mixed-State):** The classification is **Mixed-State** if Feature 5 is `Observed`, which requires that both crystalline indicators (Feature 2 & 3) and amorphous indicators (Feature 4) are simultaneously present and superimposed.

---
**REQUIRED OUTPUT FORMAT:**

Your final output **must** be a single, valid JSON object. Do not include any text, explanation, or markdown formatting before or after the JSON block. The JSON object must have exactly two keys: `classification_code` and `description`.

**JSON Structure:**
```json
{
  "classification_code": "integer",
  "description": "string"
}
```

**Instructions for each key:**
1.  **`classification_code`**: This field must contain a single integer from the set `{0, 1, 2, 3}`, corresponding to the classification you determined using the analysis framework above.
    *   `0` for Vacuum
    *   `1` for Crystalline
    *   `2` for Amorphous
    *   `3` for Mixed-State

2.  **`description`**: In a single, objective paragraph (as a string), provide a detailed analysis of the key visual features in the CBED pattern that justify your classification. Your description must be a rigorous synthesis of your findings from the internal analysis, using precise materials science terminology.
    *   **For all states**, comment on the characteristics of the central transmitted beam.
    *   **If crystalline features are present (Crystalline or Mixed-State)**, describe the **sharp, point-like Bragg diffraction spots**, noting their arrangement into a distinct geometric pattern that exhibits **clear crystallographic point group symmetry**, which is direct evidence of long-range periodic atomic order.
    *   **If amorphous features are present (Amorphous or Mixed-State)**, describe the primary feature as a **broad, diffuse intensity ring or halo**, noting that this is the hallmark of short-range atomic order. Specify that this feature can appear either as a **(a) smooth, continuous band** OR as a **(b) ring of grainy intensity fluctuations (speckles)**. This speckled appearance (case b) is the characteristic result of probing an amorphous/disordered structure with a highly coherent, nano-sized electron beam. Critically distinguish this from a crystalline state by noting the complete **absence of sharp, point-like Bragg peaks** and the lack of any **crystallographic point group symmetry** (i.e., the pattern has only radial symmetry).
    *   **For a Mixed-State classification**, explicitly state that the defining characteristic is the **superposition** of sharp, crystallographically-arranged Bragg spots on top of the diffuse (smooth or speckled) amorphous halo, providing clear evidence for the coexistence of both long-range and short-range ordered domains.
    *   **For a Vacuum classification**, state that the pattern consists solely of the central beam against a dark, featureless background.
"""

        # Process all patterns in a single batch so all photos are generated at once
        total_patterns = len(rows)
        batch_size = total_patterns
        logger.info(
            f"Processing {total_patterns} patterns in a single batch of size {batch_size}"
        )

        for batch_idx in range(0, total_patterns, batch_size):
            batch_end = min(batch_idx + batch_size, total_patterns)
            batch_rows = rows[batch_idx:batch_end]
            batch_num = (batch_idx // batch_size) + 1

            logger.info(
                f"Processing batch {batch_num}/{(total_patterns + batch_size - 1) // batch_size} ({len(batch_rows)} patterns)"
            )

            for i, row in enumerate(batch_rows):
                global_idx = batch_idx + i
                pattern_id = row.cells["id"]
                source_mat_id = row.cells["source_mat_id"]
                col_index = row.cells["col_index"]

                # Try to get existing LLM analysis for this pattern first
                analysis_query = """
                    SELECT llm_detailed_features
                    FROM llm_analysis_results
                    WHERE pattern_id = %s
                """

                analysis_rows = await sql_driver.execute_query(
                    analysis_query, [pattern_id]
                )

                if analysis_rows and analysis_rows[0].cells["llm_detailed_features"]:
                    # Use existing analysis
                    detailed_features = analysis_rows[0].cells["llm_detailed_features"]
                    if isinstance(detailed_features, str):
                        try:
                            detailed_features = json.loads(detailed_features)
                        except json.JSONDecodeError:
                            detailed_features = {}

                    pattern_descriptions.append(
                        {
                            "classification_code": detailed_features.get(
                                "classification_code", 0
                            ),
                            "description": detailed_features.get(
                                "description",
                                f"Pattern {global_idx + 1} from cluster {cluster_index}",
                            ),
                        }
                    )
                    logger.debug(f"Pattern {global_idx + 1}: Using cached analysis")
                else:
                    # Need to generate new LLM analysis
                    logger.info(f"Pattern {global_idx + 1}: Analyzing with LLM")

                    # Check if orchestrator is available for real LLM analysis
                    if not self.orchestrator:
                        logger.warning(f"No LLM orchestrator available")
                        pattern_descriptions.append(
                            {
                                "classification_code": cluster_index % 4,
                                "description": f"Pattern {global_idx + 1}: LLM unavailable, using fallback",
                            }
                        )
                        continue

                    try:
                        # Step 2: Extract image from .mat file
                        # First get the mat file path
                        mat_query = """
                            SELECT file_path, row_index
                            FROM raw_mat_files
                            WHERE id = %s
                        """
                        mat_rows = await sql_driver.execute_query(
                            mat_query, [source_mat_id]
                        )

                        if not mat_rows:
                            logger.error(
                                f"Mat file not found for source_mat_id={source_mat_id}"
                            )
                            pattern_descriptions.append(
                                {
                                    "classification_code": cluster_index % 4,
                                    "description": f"Pattern {global_idx + 1}: Mat file not found",
                                }
                            )
                            continue

                        mat_file_path = mat_rows[0].cells["file_path"]
                        row_index = mat_rows[0].cells[
                            "row_index"
                        ]  # This is what was called group_number

                        # Extract and save the image
                        try:
                            # Load mat file and extract the specific pattern
                            import scipy.io
                            import numpy as np
                            from PIL import Image as PILImage

                            mat_data = scipy.io.loadmat(mat_file_path)
                            data = mat_data["data"]
                            image = data[col_index - 1]  # col_index is 1-based

                            # Preprocess image using the same function from server
                            image = self._preprocess_image(image)

                            # Save as PNG
                            temp_image_path = os.path.join(
                                temp_images_dir,
                                f"cluster_{cluster_index}_pattern_{global_idx + 1}_{pattern_id}.png",
                            )

                            # Ensure directory exists before saving
                            os.makedirs(temp_images_dir, exist_ok=True)

                            img_uint8 = (image * 255).astype(np.uint8)
                            pil_img = PILImage.fromarray(img_uint8)
                            pil_img.save(temp_image_path)

                            # Step 3: Call LLM analysis
                            llm_result = await self.orchestrator.analyze_single_image(
                                temp_image_path, JSON_PROMPT_TEMPLATE, model
                            )

                            # Step 4: Use real LLM results
                            if (
                                isinstance(llm_result, dict)
                                and "error" not in llm_result
                            ):
                                classification_code = llm_result.get(
                                    "classification_code", cluster_index % 4
                                )
                                description = llm_result.get(
                                    "description",
                                    f"Pattern {global_idx + 1} analyzed by LLM",
                                )

                                pattern_descriptions.append(
                                    {
                                        "classification_code": classification_code,
                                        "description": description,
                                    }
                                )

                                logger.info(
                                    f"Pattern {global_idx + 1}: LLM analysis complete (code={classification_code})"
                                )

                                # Store result in database for future use
                                try:
                                    # Store result in database with all required fields
                                    store_query = """
                                        INSERT INTO llm_analysis_results (
                                            pattern_id, scan_id, clustering_run_id, cluster_id,
                                            row_index, col_index, cluster_index, llm_detailed_features
                                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                        ON CONFLICT (pattern_id) DO UPDATE SET
                                            scan_id = EXCLUDED.scan_id,
                                            clustering_run_id = EXCLUDED.clustering_run_id,
                                            cluster_id = EXCLUDED.cluster_id,
                                            row_index = EXCLUDED.row_index,
                                            col_index = EXCLUDED.col_index,
                                            cluster_index = EXCLUDED.cluster_index,
                                            llm_detailed_features = EXCLUDED.llm_detailed_features
                                    """
                                    await sql_driver.execute_query(
                                        store_query,
                                        [
                                            pattern_id,
                                            scan_id,
                                            clustering_run_id,
                                            cluster_id,
                                            row_index,
                                            col_index,
                                            cluster_index,
                                            json.dumps(llm_result),
                                        ],
                                    )
                                except Exception as store_error:
                                    logger.warning(
                                        f"Failed to store result for pattern {pattern_id}: {store_error}"
                                    )
                            else:
                                logger.error(
                                    f"Pattern {global_idx + 1}: LLM analysis failed"
                                )
                                pattern_descriptions.append(
                                    {
                                        "classification_code": cluster_index % 4,
                                        "description": f"Pattern {global_idx + 1}: LLM analysis failed",
                                    }
                                )

                        except Exception as img_error:
                            logger.error(
                                f"Pattern {global_idx + 1}: Image extraction failed - {img_error}"
                            )
                            pattern_descriptions.append(
                                {
                                    "classification_code": cluster_index % 4,
                                    "description": f"Pattern {global_idx + 1}: Image extraction failed",
                                }
                            )

                    except Exception as e:
                        logger.error(f"Pattern {global_idx + 1}: Analysis failed - {e}")
                        pattern_descriptions.append(
                            {
                                "classification_code": cluster_index % 4,
                                "description": f"Pattern {global_idx + 1}: Analysis failed",
                            }
                        )

            logger.info(f"Batch {batch_num} complete")

        logger.info(
            f"All {len(pattern_descriptions)} patterns processed for cluster {cluster_index}"
        )

        for i, row in enumerate(rows):
            pattern_id = row.cells["id"]
            source_mat_id = row.cells["source_mat_id"]
            col_index = row.cells["col_index"]

            logger.info(
                f"Processing pattern {i + 1}/16: pattern_id={pattern_id}, source_mat_id={source_mat_id}, col_index={col_index}"
            )

            # Try to get existing LLM analysis for this pattern first
            analysis_query = """
                SELECT llm_detailed_features
                FROM llm_analysis_results
                WHERE pattern_id = %s
            """

            analysis_rows = await sql_driver.execute_query(analysis_query, [pattern_id])

            if analysis_rows and analysis_rows[0].cells["llm_detailed_features"]:
                # Use existing analysis
                detailed_features = analysis_rows[0].cells["llm_detailed_features"]
                if isinstance(detailed_features, str):
                    try:
                        detailed_features = json.loads(detailed_features)
                    except json.JSONDecodeError:
                        detailed_features = {}

                pattern_descriptions.append(
                    {
                        "classification_code": detailed_features.get(
                            "classification_code", 0
                        ),
                        "description": detailed_features.get(
                            "description",
                            f"Pattern {i + 1} from cluster {cluster_index}",
                        ),
                    }
                )
                logger.info(f"Using existing LLM analysis for pattern {pattern_id}")
            else:
                # Need to generate new LLM analysis
                logger.info(
                    f"No existing analysis found for pattern {pattern_id}, generating new analysis"
                )

                # Check if orchestrator is available for real LLM analysis
                if not self.orchestrator:
                    logger.warning(
                        f"No LLM orchestrator available, using fallback for pattern {pattern_id}"
                    )
                    pattern_descriptions.append(
                        {
                            "classification_code": cluster_index % 4,
                            "description": f"Pattern {i + 1}: LLM orchestrator not available, using fallback classification",
                        }
                    )
                    continue

                try:
                    # Step 2: Extract image from .mat file
                    # First get the mat file path
                    mat_query = """
                        SELECT file_path, row_index
                        FROM raw_mat_files
                        WHERE id = %s
                    """
                    mat_rows = await sql_driver.execute_query(
                        mat_query, [source_mat_id]
                    )

                    if not mat_rows:
                        logger.error(
                            f"No mat file found for source_mat_id={source_mat_id}"
                        )
                        # Use fallback placeholder
                        pattern_descriptions.append(
                            {
                                "classification_code": cluster_index % 4,
                                "description": f"Pattern {i + 1}: Error - mat file not found for analysis",
                            }
                        )
                        continue

                    mat_file_path = mat_rows[0].cells["file_path"]
                    row_index = mat_rows[0].cells[
                        "row_index"
                    ]  # This is what was called group_number

                    logger.info(
                        f"Extracting image from mat file: {mat_file_path}, col_index={col_index}, row_index={row_index}"
                    )

                    # Extract the image using the existing function from server
                    # Import the extract function
                    import scipy.io
                    import numpy as np
                    from PIL import Image as PILImage

                    # Extract and save the image
                    try:
                        # Load mat file and extract the specific pattern
                        mat_data = scipy.io.loadmat(mat_file_path)
                        data = mat_data["data"]
                        image = data[col_index - 1]  # col_index is 1-based

                        # Preprocess image using the same function from server
                        image = self._preprocess_image(image)

                        # Save as PNG
                        temp_image_path = os.path.join(
                            temp_images_dir,
                            f"cluster_{cluster_index}_pattern_{i + 1}_{pattern_id}.png",
                        )
                        img_uint8 = (image * 255).astype(np.uint8)
                        pil_img = PILImage.fromarray(img_uint8)
                        pil_img.save(temp_image_path)

                        logger.info(f"Saved image to {temp_image_path}")

                        # Step 3: Call LLM analysis
                        logger.info(f"Calling LLM analysis for pattern {pattern_id}")
                        llm_result = await self.orchestrator.analyze_single_image(
                            temp_image_path, JSON_PROMPT_TEMPLATE, model
                        )

                        # Step 4: Use real LLM results
                        if isinstance(llm_result, dict) and "error" not in llm_result:
                            classification_code = llm_result.get(
                                "classification_code", cluster_index % 4
                            )
                            description = llm_result.get(
                                "description", f"Pattern {i + 1} analyzed by LLM"
                            )

                            pattern_descriptions.append(
                                {
                                    "classification_code": classification_code,
                                    "description": description,
                                }
                            )

                            logger.info(
                                f"Successfully analyzed pattern {pattern_id} with LLM: code={classification_code}"
                            )

                            # Optionally store the result in the database for future use
                            try:
                                # Store result in database with all required non-null fields
                                store_query = """
                                    INSERT INTO llm_analysis_results (
                                        pattern_id, scan_id, clustering_run_id, cluster_id,
                                        row_index, col_index, cluster_index, llm_detailed_features
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (pattern_id) DO UPDATE SET
                                        scan_id = EXCLUDED.scan_id,
                                        clustering_run_id = EXCLUDED.clustering_run_id,
                                        cluster_id = EXCLUDED.cluster_id,
                                        row_index = EXCLUDED.row_index,
                                        col_index = EXCLUDED.col_index,
                                        cluster_index = EXCLUDED.cluster_index,
                                        llm_detailed_features = EXCLUDED.llm_detailed_features
                                """
                                await sql_driver.execute_query(
                                    store_query,
                                    [
                                        pattern_id,
                                        scan_id,
                                        clustering_run_id,
                                        cluster_id,
                                        row_index,
                                        col_index,
                                        cluster_index,
                                        json.dumps(llm_result),
                                    ],
                                )
                                logger.info(
                                    f"Stored LLM result for pattern {pattern_id}"
                                )
                            except Exception as store_error:
                                logger.warning(
                                    f"Failed to store LLM result for pattern {pattern_id}: {store_error}"
                                )
                        else:
                            logger.error(
                                f"LLM analysis failed for pattern {pattern_id}: {llm_result}"
                            )
                            # Use fallback
                            pattern_descriptions.append(
                                {
                                    "classification_code": cluster_index % 4,
                                    "description": f"Pattern {i + 1}: LLM analysis failed, using fallback classification",
                                }
                            )

                    except Exception as img_error:
                        logger.error(
                            f"Error extracting/processing image for pattern {pattern_id}: {img_error}",
                            exc_info=True,
                        )
                        # Use fallback
                        pattern_descriptions.append(
                            {
                                "classification_code": cluster_index % 4,
                                "description": f"Pattern {i + 1}: Image extraction failed, using fallback classification",
                            }
                        )

                except Exception as e:
                    logger.error(
                        f"Error in LLM analysis pipeline for pattern {pattern_id}: {e}",
                        exc_info=True,
                    )
                    # Use fallback
                    pattern_descriptions.append(
                        {
                            "classification_code": cluster_index % 4,
                            "description": f"Pattern {i + 1}: Analysis pipeline failed, using fallback classification",
                        }
                    )

        logger.info(
            f"Generated {len(pattern_descriptions)} pattern descriptions for cluster {cluster_index}"
        )

        # Clean up temp directory
        try:
            import shutil

            shutil.rmtree(temp_images_dir, ignore_errors=True)
        except Exception:
            pass

        # Generate consensus for this cluster
        consensus_result = await self.generate_cluster_consensus(
            cluster_id, pattern_descriptions, model
        )

        # Add cluster index to result
        consensus_result["cluster_index"] = cluster_index

        logger.info(f"Completed analysis for cluster {cluster_index}")
        return consensus_result

    async def store_consensus_results(
        self,
        sql_driver: SafeSqlDriver,
        scan_id: int,
        clustering_run_id: int,
        consensus_results: List[Dict[str, Any]],
    ) -> None:
        """
        Store consensus analysis results in the database.

        Args:
            sql_driver: Database driver for storing results
            scan_id: ID of the scan
            clustering_run_id: ID of the clustering run
            consensus_results: List of consensus analysis results
        """
        logger.info(
            f"Storing consensus results for scan_id={scan_id}, clustering_run_id={clustering_run_id}"
        )
        logger.info(f"Number of results to store: {len(consensus_results)}")

        for result in consensus_results:
            logger.info(f"Storing result for cluster_index={result['cluster_index']}")

            # Insert or update llm_analyses table
            analysis_query = """
                INSERT INTO llm_analyses 
                (cluster_id, representative_patterns_count, llm_assigned_class, llm_detailed_features)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (cluster_id) 
                DO UPDATE SET 
                    representative_patterns_count = EXCLUDED.representative_patterns_count,
                    llm_assigned_class = EXCLUDED.llm_assigned_class,
                    llm_detailed_features = EXCLUDED.llm_detailed_features
            """

            # Map classification codes to class names
            class_names = {
                0: "Vacuum",
                1: "Crystalline",
                2: "Amorphous",
                3: "Mixed-State",
            }

            dominant_class = class_names.get(
                result["dominant_classification_code"], "Unknown"
            )

            # Store detailed features as JSON
            detailed_features = {
                "consensus_description": result["consensus_description"],
                "pattern_count": result["pattern_count"],
                "dominant_classification_code": result["dominant_classification_code"],
            }

            logger.info(
                f"Executing insert/update for cluster_id={result['cluster_id']}"
            )
            logger.debug(
                f"Data to store: representative_patterns_count={result['pattern_count']}, llm_assigned_class={dominant_class}"
            )

            await sql_driver.execute_query(
                analysis_query,
                [
                    result["cluster_id"],
                    result["pattern_count"],
                    dominant_class,
                    json.dumps(detailed_features),
                ],
            )

            logger.info(
                f"Stored consensus results for cluster {result['cluster_index']}"
            )

        logger.info("Completed storing all consensus results")
