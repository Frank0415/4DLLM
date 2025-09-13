"""
LLM Orchestrator for managing parallel LLM analysis tasks.
"""

import asyncio
import aiohttp
import base64
import json
import logging
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Set up unbuffered logging to /tmp/llm_logging with rotation
log_file_path = "/tmp/llm_logging"
# Check if log file exists and is larger than 10MB, if so, rotate it
if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 10 * 1024 * 1024:
    # Rotate the log file
    if os.path.exists(f"{log_file_path}.1"):
        os.remove(f"{log_file_path}.1")
    os.rename(log_file_path, f"{log_file_path}.1")

file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
# Force unbuffered output
file_handler.stream.reconfigure(line_buffering=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False  # Don't propagate to root logger


class LLMOrchestrator:
    """Orchestrator using aiohttp and REST API to manage and orchestrate LLM analysis tasks"""

    def __init__(
        self, api_keys: List[str], base_url: str, model: str = "gemini-pro"
    ):
        """
        Initialize LLM orchestrator.

        Args:
            api_keys: List of API keys for load balancing
            base_url: Base URL for the LLM API
            model: Model name to use
        """
        self.api_keys = api_keys
        self.base_url = base_url
        self.model = model
        self._key_idx = 0
        logger.info(f"Initialized LLMOrchestrator with base_url={base_url}, model={model}")

    def _get_next_key(self) -> str:
        """Cycle through API keys to distribute load"""
        key = self.api_keys[self._key_idx]
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        logger.debug(f"Using API key index {self._key_idx}")
        return key

    async def analyze_text_prompt(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """
        Analyze a text prompt using the LLM through REST API.
        
        Args:
            prompt: Text prompt for the LLM
            model: Optional model override
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting text prompt analysis")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        api_key = self._get_next_key()
        model = model or self.model
        logger.info(f"Using model: {model}")
        
        # Construct URL for Google Gemini API
        request_url = f"{self.base_url}/v1beta/models/{model}:generateContent"
        logger.info(f"Request URL: {request_url}")
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        logger.debug(f"Request headers: {headers}")
        
        # Build request body for Google Gemini API
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        }
        logger.debug(f"Payload keys: {list(payload.keys())}")
        
        async with aiohttp.ClientSession() as session:
            try:
                logger.info("Sending POST request to LLM API")
                async with session.post(
                    request_url, headers=headers, json=payload
                ) as response:
                    logger.info(f"Received response with status: {response.status}")
                    if response.status == 200:
                        response_json = await response.json()
                        logger.info("Successfully parsed JSON response")
                        logger.debug(f"Response JSON keys: {list(response_json.keys())}")
                        
                        # Extract content from Gemini API response
                        raw_response_text = (
                            response_json.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        logger.info(f"Raw response text length: {len(raw_response_text)} characters")
                        logger.debug(f"Raw response text (first 200 chars): {raw_response_text[:200]}")
                        
                        # Parse JSON response
                        logger.info("Parsing JSON from LLM response")
                        result = self._extract_json_from_response(raw_response_text)
                        logger.info(f"Parsed result keys: {list(result.keys())}")
                        
                        # Add timestamp
                        from datetime import datetime
                        result["timestamp"] = datetime.utcnow().isoformat()
                        logger.info("Added timestamp to result")
                        
                        logger.info("Completed text prompt analysis successfully")
                        return result
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
                logger.error(f"Network or request error when calling LLM REST API: {e}", exc_info=True)
                return {"error": str(e), "raw_text": ""}

    async def analyze_single_image(
        self, image_path: str, prompt: str, model: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a single image using aiohttp through REST API.

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt for the LLM
            model: Optional model override

        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting single image analysis for {image_path}")
        logger.debug(f"Image path: {image_path}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        api_key = self._get_next_key()
        model = model or self.model
        logger.info(f"Using model: {model}")

        # Construct URL for Google Gemini API
        request_url = f"{self.base_url}/v1beta/models/{model}:generateContent"
        logger.info(f"Request URL: {request_url}")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        logger.debug(f"Request headers: {headers}")

        try:
            logger.info("Reading image file and encoding to base64")
            b64_image = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            logger.info(f"Base64 encoded image length: {len(b64_image)} characters")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return {"error": f"File not found: {image_path}"}

        # Build request body for Google Gemini Vision API
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                    ]
                }
            ],
            # Add generation config to force JSON output
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        }
        logger.debug(f"Payload keys: {list(payload.keys())}")

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("Sending POST request to LLM API")
                async with session.post(
                    request_url, headers=headers, json=payload
                ) as response:
                    logger.info(f"Received response with status: {response.status}")
                    if response.status == 200:
                        response_json = await response.json()
                        logger.info("Successfully parsed JSON response")
                        logger.debug(f"Response JSON keys: {list(response_json.keys())}")
                        
                        # Extract content from Gemini API response
                        raw_response_text = (
                            response_json.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        logger.info(f"Raw response text length: {len(raw_response_text)} characters")
                        logger.debug(f"Raw response text (first 200 chars): {raw_response_text[:200]}")
                        
                        # Parse JSON response
                        logger.info("Parsing JSON from LLM response")
                        result = self._extract_json_from_response(raw_response_text)
                        logger.info(f"Parsed result keys: {list(result.keys())}")
                        
                        # Add timestamp
                        from datetime import datetime
                        result["timestamp"] = datetime.utcnow().isoformat()
                        logger.info("Added timestamp to result")
                        
                        logger.info("Completed single image analysis successfully")
                        return result
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
                logger.error(f"Network or request error when calling LLM REST API: {e}", exc_info=True)
                return {"error": str(e), "raw_text": ""}

    async def analyze_batch(
        self,
        image_paths: List[str],
        prompt: str,
        semaphore: asyncio.Semaphore,
        model: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of images with rate limiting.

        Args:
            image_paths: List of image paths to analyze
            prompt: Analysis prompt for the LLM
            semaphore: Semaphore for rate limiting
            model: Optional model override

        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis of {len(image_paths)} images")
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(
                self._analyze_with_semaphore(image_path, prompt, semaphore, model)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Completed batch analysis with {len(results)} results")
        return results

    async def _analyze_with_semaphore(
        self,
        image_path: str,
        prompt: str,
        semaphore: asyncio.Semaphore,
        model: str = None,
    ) -> Dict[str, Any]:
        """Analyze single image with semaphore for rate limiting."""
        logger.info(f"Analyzing image with semaphore: {image_path}")
        async with semaphore:
            result = await self.analyze_single_image(image_path, prompt, model)
            # Add timestamp if not present
            if "timestamp" not in result:
                from datetime import datetime
                result["timestamp"] = datetime.utcnow().isoformat()
            logger.info(f"Completed semaphore analysis for: {image_path}")
            return result

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract a JSON object from potentially noisy LLM response text.
        """
        logger.info("Starting JSON extraction from LLM response")
        logger.debug(f"Input text length: {len(text) if text else 0} characters")
        
        if not isinstance(text, str):
            logger.warning("Invalid input type for JSON extraction, expected string")
            return {
                "error": "Invalid input type, expected string.",
                "raw_text": str(text),
            }

        # Try to find JSON enclosed in ```json ... ``` markdown blocks first
        logger.info("Searching for JSON in markdown code blocks")
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            clean_text = match.group(1)
            logger.info("Found JSON in markdown code blocks")
        else:
            logger.info("No markdown code blocks found, searching for JSON object")
            # If no markdown block found, try to find the first valid JSON object
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                clean_text = text[start_index : end_index + 1]
                logger.info("Found JSON object by bracket matching")
            else:
                clean_text = text
                logger.warning("No JSON object found, using raw text")

        logger.debug(f"Clean text length: {len(clean_text)} characters")
        logger.debug(f"Clean text (first 200 chars): {clean_text[:200]}")
        
        try:
            result = json.loads(clean_text)
            logger.info("Successfully parsed JSON")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.warning(
                f"Unable to parse LLM output as JSON. Raw text: '{text[:100]}...'"
            )
            return {"error": "Failed to decode JSON.", "raw_text": text}
