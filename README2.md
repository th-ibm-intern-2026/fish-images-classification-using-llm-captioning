# Repository Reference Guide (API-focused)

This document lists and explains the HTTP API routes implemented in `BE/api_services.py`. It describes the request/response shapes, behavior, environment variables used, and notes about how embedding and search are wired.

**API Reference (BE/api_services.py)**

This section documents the Flask API implemented in `BE/api_services.py`. The service exposes endpoints for captioning images (WatsonX), searching by embedding, image identification, and generation (conversational responses). The routes assume environment variables (see `env-template`) are set for Watsonx, Elasticsearch, and IBM COS credentials.

- **GET /live**
  - **Method:** GET
  - **Purpose:** Healthcheck endpoint; returns basic readiness status.
  - **Request:** none
  - **Response:** `200 OK` JSON: `{"status": "ok"}`

- **POST /search**
  - **Method:** POST
  - **Purpose:** Accepts free-text (or an image caption) and returns the top-N fish by vector similarity.
  - **Request JSON:** `{"text": "<description or caption>", "n": 5}` (only `text` is required; `n` is optional)
  - **Behavior:**
    - Uses `EmbeddingService.embed_text(...)` to convert the provided `text` into an embedding.
    - Calls `ElasticsearchQuery.search_embedding(...)` against the index configured in the module (`index_name`) using field `physical_description_embedding`.
    - Formats hits via `function.return_top_n_fish_simple(...)` and returns results.
  - **Response:** `200 OK` JSON: `{"input": "...", "results": [{"fish_name": "...", "thai_fish_name": "...", "scientific_name": "...", "order_name": "...", "score": 1.23}, ...]}`
  - **Errors:** Returns `503` with a fallback JSON if embedding service, Elasticsearch, or other internal errors occur.
  - **Notes:** This endpoint performs embedding at runtime; you can pass a caption returned by `/image_captioning` to this endpoint to find matching fish.

- **POST /image_captioning**
  - **Method:** POST
  - **Purpose:** Caption an image stored in IBM Cloud Object Storage (COS) using Watsonx vision models.
  - **Request JSON:** `{"image": "<cos-object-key>"}` — the path/key in the COS bucket (the code uses bucket `fish-image-bucket`).
  - **Behavior:**
    - Fetches image bytes from COS using `ibm_boto3` with `IBM_COS_API_KEY`, `IBM_COS_RESOURCE_INSTANCE_ID`, and `IBM_COS_ENDPOINT`.
    - Converts image to base64 and calls `get_fish_description_from_watsonxai(...)` in `watsonx_captioning.py`.
    - Returns the caption string from Watsonx.
  - **Response:** `200 OK` JSON: `{"caption": "<generated caption text>"}`
  - **Errors:** If COS fetch fails or Watsonx fails, returns `503` with a fallback structure: `{"error": "image_captioning service unavailable", "fallback": True, "details": "..."}`
  - **Notes:** This endpoint does NOT automatically embed the caption and search the index. To get matches, call `/search` with the caption text or extend this endpoint to call `embed_text` + `search_embedding` before returning.

- **POST /image_identification**
  - **Method:** POST
  - **Purpose:** Returns a structured JSON describing whether the image contains a fish and details (Thai name, scientific name, physical description, habitat).
  - **Request JSON:** `{"image": "<cos-object-key>"}`
  - **Behavior:** Similar to `/image_captioning` but uses `get_json_generated_image_details(...)` to request a JSON-only response from Watsonx that includes `image_contains_fish` and `fish_details`.
  - **Response:** `200 OK` JSON example: `{"image_contains_fish": true, "fish_details": {"fish_name": "...", "scientific_name": "...", "physical_description": "...", "habitat": "..."}}`
  - **Errors:** Returns `503` with fallback payload on Watsonx/COS errors.

- **POST /generation**
  - **Method:** POST
  - **Purpose:** General-purpose generative Q&A powered by Watsonx; accepts an optional `context` and `chat_history` to keep continuity.
  - **Request JSON:** `{"question": "<text>", "chat_history": [ {"role": "user|assistant", "content": "..."}, ... ], "context": "<optional context string>"}`
  - **Behavior:**
    - If `context` is present, uses `get_generated_response_with_context` to include it in the system prompt.
    - Otherwise calls `get_generated_response` which internally embeds the `question`, finds similar fish references (both physical and general) via `ElasticsearchQuery.search_embedding(...)`, and builds a reference block included in the chat system prompt before calling Watsonx.
  - **Response:** `200 OK` JSON: `{"response": "<model-generated text>"}`
  - **Errors:** Returns `503` with fallback payload in case of model/service errors.
  - **Notes:** This endpoint already performs embedding of the user's question to gather reference documents from Elasticsearch and uses those references to improve accuracy.

- **POST /search_with_scientific_name**
  - **Method:** POST
  - **Purpose:** Lookup a single fish by `scientific_name` using text search.
  - **Request JSON:** `{"scientific_name": "Arothron hispidus"}`
  - **Behavior:** Executes `ElasticsearchQuery.search_text(...)` on the `scientific_name` field and returns the matching document(s) formatted by `function.return_fish_info(...)`.
  - **Response:** `200 OK` JSON: `{"scientific_name": "...", "fish_data": [...], "message": "Success"}`
  - **Notes:** Designed for exact-name lookups; returns a helpful message when not found.

**Implementation notes / env-vars used by API**
- Watsonx: `WATSONX_APIKEY`, `IBM_WATSONX_AI_INFERENCE_URL`, `PROJECT_ID`, `IAM_IBM_CLOUD_URL` are used by `watsonx_captioning.py` and `generation.py`.
- Embedding: `EMBEDDING_SERVICE_URL` if `EmbeddingService` is configured to call a remote endpoint (or local sentence-transformer model otherwise).
- Elasticsearch: `es_endpoint`, `es_username`, `es_password`, and `es_cert_path` used by `ElasticsearchQuery`.
- COS: `IBM_COS_API_KEY`, `IBM_COS_RESOURCE_INSTANCE_ID`, `IBM_COS_ENDPOINT` used to fetch images.

**How to wire caption → search automatically**
- Option A (client): Call `/image_captioning` to get caption, then call `/search` with the returned caption.
- Option B (server): Modify `/image_captioning` to call `EmbeddingService.embed_text(caption)` then `esq.search_embedding(...)` and return both `caption` and `results` in one response. The code path in `BE/main.py` shows the exact sequence used in a local example.

---

**POST /search_possible_fish**
  - **Method:** POST
  - **Purpose:** High-level image-based candidate finder that returns the AI's full response including candidate fish, scores, and reasoning. This route integrates image fetch from IBM COS, Watsonx inference, and returns the structured result produced by the model (it does not itself perform additional embedding/search against Elasticsearch).
  - **Request JSON:** `{"image": "<cos-object-key>"}` where `<cos-object-key>` is the object key inside the `fish-image-bucket` bucket.
  - **Behavior:**
    - Fetches the image from IBM COS (requires `IBM_COS_API_KEY`, `IBM_COS_RESOURCE_INSTANCE_ID`, `IBM_COS_ENDPOINT`).
    - Encodes the image to base64 and calls the internal `identify_fish_candidates(...)` helper which drives Watsonx to return a JSON object with `top_candidates`, `scores`, and `reasons`.
    - Returns the full AI response JSON directly to the client.
  - **Response:** `200 OK` JSON — the AI-generated JSON structure. Example keys: `{"image_contains_fish": true, "top_candidates": [{"fish_name": "...", "score": 0.98, "reason": "..."}, ...], "raw_ai_output": {...}}` (actual schema may vary depending on model prompt and post-processing).
  - **Errors:** Returns `500` with a fallback payload on COS or Watsonx errors.
  - **Notes:** This endpoint is useful if you want the raw candidate set and model reasoning. It intentionally returns the AI output rather than performing a second embedding/search step. To convert the returned candidate names into indexed, embedded matches (for score comparison with your Elasticsearch index), see the CSV update and ingestion notes below.
  - **Candidate Source (Important):** The set of fish names the model will tend to surface is constrained by your curated list in `BE/Marine_Fish_Possible_Output.csv`. If you want additional species to appear in `/search_possible_fish` results, append new rows there. Each row format: `Fish Name,Physical Description`. Keep descriptions concise but distinctive (color, shape, markings) — they feed into prompting quality.

---

Adding new fish entries (append to CSV and re-ingest)
- Source CSV: `EXTRACTION/DATA/fish-description-files/Marine_Fish_Species_Formatted_updated.csv` is the canonical dataset used for ingestion. Its header columns are:
  - `Fish Name, Thai Fish Name, Scientific Name, Order Name, General Description, Physical Description, habitat, Avg Length(cm), Avg Age(years), Avg DepthLevel(m), Avg Weight(kg)`
- If you want the AI's `top_candidates` (from `/search_possible_fish`) to be present in the Elasticsearch-backed index and searchable by embedding, do the following:
  1. Append a new row to `Marine_Fish_Species_Formatted_updated.csv` using the same column order. Provide as much metadata as possible (common name, Thai name, scientific name, general and physical descriptions). Physical description is especially valuable for matching by appearance.
  2. If you want automated physical description generation, you can use scripts in `EXTRACTION/` (for example `create_embedding_csv.py` and `physical_description_service.py`) to generate or refine `Physical Description` values.
  3. After updating the CSV, run the ingestion pipeline: `INGESTION/main.py`. This script will:
     - Load the updated CSV
     - Call `EmbeddingService.embed_text(...)` on both `General Description` and `Physical Description` to create dense vectors
     - Add `general_description_embedding` and `physical_description_embedding` columns to the dataframe
     - Bulk upload documents into the Elasticsearch index (`index_name` configured in the script)
  4. Once ingestion completes, the new fish entries will be available for embedding-based search (e.g., `/search` and the `generation` endpoint's internal reference lookup).

- Notes & tips:
  - Keep backups of the CSV before large edits. Use proper UTF-8 encoding when adding Thai names or non-ASCII text.
  - If you only need a few quick entries for testing, you can directly add them to the index using `INGESTION/elasticsearch_manager.py` utilities (e.g., via a small script that constructs documents and calls `bulk(...)`). However, using the CSV + ingestion flow ensures consistent fields and embeddings.
  - After ingesting, you can verify presence with `ElasticsearchManager.get_index_info(index_name)` or by using `BE/main.py` example flow to embed a test caption and run `esq.search_embedding(...)`.


---

**GET /isGemini**
  - **Method:** GET
  - **Purpose:** Check whether the service is currently using the Gemini model for `/search_possible_fish`.
  - **Request:** none
  - **Behavior:** Returns the current toggle state from the in-process flag `USE_GEMINI`.
  - **Response:** `200 OK` JSON: `{"USE_GEMINI": true|false}`
  - **Notes:** This only affects the model used inside `/search_possible_fish`. Other endpoints continue to use Watsonx.

**GET /changeModel**
  - **Method:** GET
  - **Purpose:** Toggle the model used by `/search_possible_fish` between Watsonx (default) and Gemini.
  - **Request:** none
  - **Behavior:** Flips the global in-memory flag `USE_GEMINI`. If it was `false` (Watsonx), it becomes `true` (Gemini), and vice versa.
  - **Response:** `200 OK` JSON: `{"USE_GEMINI": true|false}` reflecting the new state after the toggle.
  - **Notes:** This toggle is process-local and non-persistent. It resets on app restart. Requires `GEMINI_API_KEY` to be set in the environment for Gemini to work.

---