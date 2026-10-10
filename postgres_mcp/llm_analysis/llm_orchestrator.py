import asyncio
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


class LLMOrchestrator:
    def __init__(self, api_keys, base_url, model='gemini-pro'):
        if not api_keys:
            raise ValueError('Configure at least one LLM API key')
        self.api_keys = api_keys
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._key_idx = 0

    def _get_next_key(self):
        key = self.api_keys[self._key_idx]
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        return key

    async def _request(self, parts, model=None):
        url = f'{self.base_url}/v1beta/models/{model or self.model}:generateContent'
        payload = {'contents': [{'parts': parts}],
                   'generationConfig': {'response_mime_type': 'application/json', 'temperature': 0.0}}
        headers = {'x-goog-api-key': self._get_next_key()}
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        return {'error': f'LLM HTTP {response.status}', 'raw_text': await response.text()}
                    data = await response.json()
                    candidates = data.get('candidates') or []
                    if not candidates:
                        return {'error': 'LLM returned no candidate', 'raw_text': json.dumps(data)}
                    text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', []))
                    result = self._extract_json_from_response(text)
                    result['timestamp'] = datetime.now(timezone.utc).isoformat()
                    return result
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                return {'error': str(error)}

    async def analyze_text_prompt(self, prompt, model=None):
        return await self._request([{'text': prompt}], model)

    async def analyze_single_image(self, image_path, prompt, model=None):
        try:
            content = await asyncio.to_thread(Path(image_path).read_bytes)
        except OSError as error:
            return {'error': str(error)}
        return await self._request([
            {'text': prompt},
            {'inline_data': {'mime_type': 'image/png', 'data': base64.b64encode(content).decode('ascii')}},
        ], model)

    async def analyze_batch(self, image_paths, prompt, semaphore, model=None):
        return await asyncio.gather(*(self._analyze_with_semaphore(path, prompt, semaphore, model)
                                      for path in image_paths))

    async def _analyze_with_semaphore(self, image_path, prompt, semaphore, model=None):
        async with semaphore:
            return await self.analyze_single_image(image_path, prompt, model)

    def _extract_json_from_response(self, text):
        start, end = text.find('{'), text.rfind('}')
        try:
            value = json.loads(text[start:end + 1] if start >= 0 and end > start else text)
            if not isinstance(value, dict):
                raise ValueError('LLM response must be a JSON object')
            return value
        except (ValueError, TypeError):
            return {'error': 'LLM returned invalid JSON', 'raw_text': text}
