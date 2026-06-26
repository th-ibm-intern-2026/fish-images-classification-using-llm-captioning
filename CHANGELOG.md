# Changelog

## Unreleased — AI rerank/gate pipeline

### Added
- `BE/anthropic_captioning.py`, `BE/deepseek_captioning.py`, `BE/text_rerank.py` — Anthropic (Claude) vision captioning/identification, the Haiku non-fish gate, and the shared text-only rerank logic (DeepSeek primary, Haiku fallback).
- Provider management in `api_services.py`: `available_providers()` / `ACTIVE_PROVIDER` / `RERANK_PROVIDER`, conditional client construction (a missing key no longer crashes startup), `/changeModel` (cycles configured providers) and `/currentModel`.
- `fish_constants.py`: `PHYSICAL_FEATURE_FIELDS`, the 7-field caption/rerank prompts (`CAPTION_MATCH_*`, `RERANK_SYSTEM`) and a structured `SYSTEM_CONTENT_SINGLE`.
- `env-template`: `IBM_COS_BUCKET`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `DEEPSEEK_API_KEY`, `RERANK_PROVIDER`, `SPACE_ID`, `IBM_CLOUD_URL`.

### Changed
- `/identify_and_search` is now the primary identification endpoint: Haiku gate -> caption -> Elasticsearch kNN (top-10) -> DeepSeek/Haiku rerank, returning the rich response (`reranked_results`, `observed_features`, `image_contains_fish`, `rejection_reason`). Image input remains a **COS object key** (fetched via the shared `fetch_image_from_cos()` helper).
- `/search_possible_fish`, `/image_identification`, `/image_captioning` now branch on `ACTIVE_PROVIDER` (Anthropic/Gemini/WatsonX/Groq) while keeping the COS-key entry point.
- `requirements.txt`: added `anthropic`; removed a duplicate `ibm-cos-sdk` line.
- `EXTRACTION/`: synced to the working set (removed the deprecated `Marine_Fish_Species_Full_Description.*` files incl. the `.xlsx`; `requirement.txt` now lists the packages actually imported — `pandas`, `python-dotenv`, `anthropic`).

### Notes
- The gate + rerank run when `ACTIVE_PROVIDER == anthropic` (needs `ANTHROPIC_API_KEY`, plus `DEEPSEEK_API_KEY` for the preferred rerank); otherwise the service degrades to Elasticsearch-only results.
- COS bucket defaults to `fish-image-bucket` (`IBM_COS_BUCKET`); keep this aligned with the frontend's upload bucket.
