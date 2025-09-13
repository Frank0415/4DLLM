"""
LLM Orchestrator for managing parallel LLM analysis tasks.
"""

import asyncio
import aiohttp
import base64
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Orchestrator using aiohttp and REST API to manage and orchestrate LLM analysis tasks"""

    def __init__(
        self, api_keys: List[str], base_url: str, model: str = "default-model"
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

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt for the LLM
            model: Optional model override

        Returns:
            Dictionary containing analysis results
        """
        api_key = self._get_next_key()
        model = model or self.model

        # Construct URL for the LLM API
        request_url = f"{self.base_url}/v1/models/{model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            b64_image = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return {"error": f"File not found: {image_path}"}

        # Build request body for generic LLM API
        payload = {
            "prompt": prompt,
            "image": b64_image,
            "format": "json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    request_url, headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        # Extract content from API response
                        raw_response_text = response_json.get("content", "")
                        logger.info(f"LLM response: {raw_response_text[:100]}...")
                        # Parse JSON response
                        result = self._extract_json_from_response(raw_response_text)
                        # Add timestamp
                        from datetime import datetime
                        result["timestamp"] = datetime.utcnow().isoformat()
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
                logger.error(f"Network or request error when calling LLM REST API: {e}")
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
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(
                self._analyze_with_semaphore(image_path, prompt, semaphore, model)
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _analyze_with_semaphore(
        self,
        image_path: str,
        prompt: str,
        semaphore: asyncio.Semaphore,
        model: str = None,
    ) -> Dict[str, Any]:
        """Analyze single image with semaphore for rate limiting."""
        async with semaphore:
            result = await self.analyze_single_image(image_path, prompt, model)
            # Add timestamp if not present
            if "timestamp" not in result:
                from datetime import datetime
                result["timestamp"] = datetime.utcnow().isoformat()
            return result

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract a JSON object from potentially noisy LLM response text.
        """
        if not isinstance(text, str):
            return {
                "error": "Invalid input type, expected string.",
                "raw_text": str(text),
            }

        # Try to find JSON enclosed in ```json ... ``` markdown blocks first
        import re

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
            logger.warning(
                f"Unable to parse LLM output as JSON. Raw text: '{text[:100]}...'"
            )
            return {"error": "Failed to decode JSON.", "raw_text": text}
