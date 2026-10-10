# LLM analysis

`LLMOrchestrator` makes bounded Gemini-compatible `generateContent` requests. It validates JSON output and returns provider failures as errors. `analyze_batch(image_paths, prompt, semaphore, model=None)` limits concurrency using the supplied semaphore.

`AnalysisPipeline` resolves database cluster IDs, samples patterns across scan coordinates, reads their registered MAT paths, prepares temporary images, and stores successful responses. Temporary directories belong to each pipeline and are cleaned after use. It can target a specific clustering run.

Classification codes are 0 vacuum, 1 crystalline, 2 amorphous, and 3 mixed. Invalid responses are failures; no classification is inferred from a cluster number. The cluster suggestion uses the most frequent successful sample classification, while original sample responses, model, and prompt remain in the record.

`ConsensusAnalyzer` reuses the same image analysis and asks the LLM for shared observations across the samples. Consensus failures remain errors.

`llm_analyses` stores the latest cluster suggestion; `llm_representative_patterns` identifies samples; `llm_analysis_results` stores the latest per-pattern response; tags reference the result record ID. Human labels live in a separate table. Browser task reports retain earlier returned samples when suggestions are updated.

Configure `api_keys`, `base_url`, and `model` in `config/api_keys.json` or browser Settings. Import, preview, and clustering do not require LLM credentials.
