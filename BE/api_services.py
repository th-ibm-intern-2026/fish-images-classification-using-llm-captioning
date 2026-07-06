from flask import Flask, request, jsonify
from watsonx_captioning import convert_image_to_base64, get_fish_description_from_watsonxai, get_json_generated_image_details
from elasticsearch_query import ElasticsearchQuery
from embedding_service import EmbeddingService
from function import return_top_n_fish, return_top_n_fish_simple, return_fish_info
from generation import get_generated_response, get_generated_response_with_context
import os
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config
import io
import logging
import base64
import traceback
from fish_services import get_watsonx_token, identify_fish_candidates


load_dotenv()
COS_BUCKET_NAME = os.environ.get('IBM_COS_BUCKET_NAME', 'fish-image-bucket')
es_endpoint = os.environ["es_endpoint"]
es_username = os.environ["es_username"]
es_password = os.environ["es_password"]

def get_cos_client():
    """Create an IBM COS client using HMAC credentials (same as Hono API)."""
    access_key = os.environ.get('IBM_COS_ACCESS_KEY_ID')
    secret_key = os.environ.get('IBM_COS_SECRET_ACCESS_KEY')
    auth_endpoint = os.environ.get('IBM_AUTH_ENDPOINT', '')
    endpoint_url = auth_endpoint if auth_endpoint.startswith('https://') else f'https://{auth_endpoint}'
    return ibm_boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )
index_name = 'fish_index_v4'    
esq = ElasticsearchQuery(es_endpoint, es_username, es_password)
emb = EmbeddingService('watsonx')

# env
watsonx_api_key = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
ibm_cloud_iam_url = os.getenv("IAM_IBM_CLOUD_URL", None)
chat_url = os.getenv("IBM_WATSONX_AI_INFERENCE_URL", None)

app = Flask(__name__)

# Dummy fallback response
def fallback_response(service_name, error_msg=None):
    resp = {"error": f"{service_name} service unavailable", "fallback": True}
    if error_msg:
        resp["details"] = error_msg
    return resp

@app.route("/live", methods=["GET"])
def live():
    return jsonify(status="ok"), 200


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        text_input = data.get("text", "")
        if not text_input:
            return jsonify({"error": "No text input provided"}), 400

        caption_embedding = emb.embed_text(text_input)
        hits = esq.search_embedding(index_name=index_name, embedding_field='physical_description_embedding', query_vector=caption_embedding, size=5)
        # top_n_fish = return_top_n_fish(hits, n=5)
        top_n_fish = return_top_n_fish_simple(hits, n=5)
        return jsonify({"input": text_input, "results": top_n_fish})
    except Exception as e:
        print(f"Error in search: {e}")
        traceback.print_exc()
        app.logger.error(f"Error in search: {e}")
        return jsonify(fallback_response("search", f"error: {e} data {data}")), 503

# This service might take a while to respond due to image processing
@app.route("/image_captioning", methods=["POST"])
def image_captioning():
    try:
        data = request.get_json()
        image = data.get("image", "")
        app.logger.info(f"Received image: {image}")
        if not image:
            app.logger.error("No image provided in request")
            return jsonify({"error": "No image provided"}), 400

        # COS fetch block
        try:
            app.logger.info("Fetching image from COS")
            cos = get_cos_client()
            response = cos.get_object(Bucket=COS_BUCKET_NAME, Key=image)
            image_bytes = response['Body'].read()
            pic_string = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as cos_e:
            traceback.print_exc()
            app.logger.error(f"COS fetch error: {cos_e}")
            return jsonify(fallback_response("image_captioning", f"COS fetch error: {cos_e}")), 503

        # WatsonX call block
        try:
            app.logger.info("Calling WatsonX for image captioning")
            caption = get_fish_description_from_watsonxai(pic_string)
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"WatsonX error: {ai_e}")
            return jsonify(fallback_response("image_captioning", f"WatsonX error: {ai_e}")), 503

        return jsonify({"caption": caption})
    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unknown error in image_captioning: {e}")
        return jsonify(fallback_response("image_captioning", str(e))), 503

@app.route("/image_identification", methods=["POST"])
def image_identification():
    try:
        data = request.get_json()
        image = data.get("image", "")
        app.logger.info(f"Received image: {image}")
        if not image:
            app.logger.error("No image provided in request")
            return jsonify({"error": "No image provided"}), 400

        # COS fetch block
        try:
            app.logger.info("Fetching image from COS")
            cos = get_cos_client()
            response = cos.get_object(Bucket=COS_BUCKET_NAME, Key=image)
            image_bytes = response['Body'].read()
        except Exception as cos_e:
            traceback.print_exc()
            app.logger.error(f"COS fetch error: {cos_e}")
            return jsonify(fallback_response("image_captioning", f"COS fetch error: {cos_e}")), 503

        # Base64 conversion block
        try:
            app.logger.info("Converting image to base64")
            pic_string = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as b64_e:
            traceback.print_exc()
            app.logger.error(f"Base64 conversion error: {b64_e}")
            return jsonify(fallback_response("image_captioning", f"Base64 error: {b64_e}")), 503

        # AI call block
        try:
            app.logger.info("Calling WatsonX for image captioning")
            json_result = get_json_generated_image_details(pic_string)
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"WatsonX error: {ai_e}")
            return jsonify(fallback_response("image_captioning", f"WatsonX error: {ai_e}")), 503

        if not json_result:
            return jsonify({"error": "AI could not identify fish (Returned None)"}), 500

        print("this is json_result",json_result)
        return json_result
    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unknown error in image_captioning: {e}")
        return jsonify(fallback_response("image_captioning", str(e))), 503

@app.route("/generation", methods=["POST"])
def generation():
    try:
        data = request.get_json()
        question = data.get("question", "")
        chat_history = data.get("chat_history", [])  # List of previous messages
        context = data.get("context", "") #optioal context for the question

        if context:
            response_text = get_generated_response_with_context(question, context, chat_history)
        else:
            response_text = get_generated_response(question, chat_history)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error in image_captioning: {e}")
        traceback.print_exc()
        app.logger.error(f"Error in generation: {e}")
        return jsonify(fallback_response("generation", str(e))), 503
    
@app.route("/search_with_scientific_name", methods=["POST"])
def search_with_scientific_name():
    try:
        data = request.get_json()
        scientific_name = data.get("scientific_name", "")
        if not scientific_name:
            return jsonify({"error": "No scientific name provided"}), 400

        print(f"Searching for scientific name: {scientific_name}")
        # Use text search on the scientific_name field, only 1 result
        hits = esq.search_text(index_name=index_name, field='scientific_name', text=scientific_name, size=1)

        fish_data = return_fish_info(hits)
        if not fish_data:
            return jsonify({
                "scientific_name": scientific_name,
                "fish_data": [],
                "message": "No fish found with the given scientific name."
            }), 200
        return jsonify({
            "scientific_name": scientific_name,
            "fish_data": fish_data,
            "message": "Success"
        }), 200
    except Exception as e:
        print(f"Error in search_with_scientific_name: {e}")
        traceback.print_exc()
        app.logger.error(f"Error in search_with_scientific_name: {e}")
        return jsonify({
            "scientific_name": scientific_name if 'scientific_name' in locals() else "",
            "fish_data": [],
            "message": f"Service error: {str(e)}"
        }), 503



### Below this one
### /identify_and_search is endpoint using to validate accuracy of datad
@app.route("/identify_and_search", methods=["POST"])
def identify_and_search():
    """
    1. Fetches an image (COS path)
    2. Calls WatsonX to generate a descriptive caption.
    3. Uses the caption to perform a semantic vector search in Elasticsearch.
    4. Returns the search results.
    """
    try:
        data = request.get_json()
        image = data.get("image", "")
        if not image:
            app.logger.error("No image path provided in request for /identify_and_search")
            return jsonify({"error": "No image path (COS Key) provided"}), 400

        app.logger.info(f"Starting 2-step process for image: {image}")

        # --- STEP 1: Fetch Image and Encode ---
        try:
            app.logger.info("Fetching image from COS.")
            cos = get_cos_client()
            response = cos.get_object(Bucket=COS_BUCKET_NAME, Key=image)
            image_bytes = response['Body'].read()
            pic_string = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as cos_e:
            traceback.print_exc()
            app.logger.error(f"COS/Base64 error in identify_and_search: {cos_e}")
            return jsonify(fallback_response("identify_and_search (Image Load)", f"Image load error: {cos_e}")), 503

        # --- STEP 2: WatsonX Image Captioning (Call get_fish_description_from_watsonxai) ---
        try:
            app.logger.info("Calling WatsonX for image captioning (Pass 1)")
            # Note: Using the simple captioning function first
            caption = get_fish_description_from_watsonxai(pic_string)
            app.logger.info(f"Generated Caption: {caption[:100]}...")
            if not caption:
                return jsonify({"error": "AI failed to generate a caption"}), 500
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"WatsonX Captioning error in identify_and_search: {ai_e}")
            return jsonify(fallback_response("identify_and_search (WatsonX Captioning)", f"AI error: {ai_e}")), 503

        # --- STEP 3: Semantic Search using Caption (Code copied from /search) ---
        try:
            app.logger.info("Starting Elasticsearch vector search with the generated caption.")
            text_input = caption
            caption_embedding = emb.embed_text(text_input)
            hits = esq.search_embedding(index_name=index_name, embedding_field='physical_description_embedding', query_vector=caption_embedding, size=5)
            # top_n_fish = return_top_n_fish(hits, n=5) # Use the simpler return function for consistency
            top_n_fish = return_top_n_fish_simple(hits, n=5)
            
            # --- FINAL RESPONSE ---
            return jsonify({
                "input_image": image,
                "ai_generated_caption": caption,
                "elasticsearch_results": top_n_fish
            })
        except Exception as es_e:
            traceback.print_exc()
            app.logger.error(f"Elasticsearch search error in identify_and_search: {es_e}")
            return jsonify(fallback_response("identify_and_search (Elasticsearch Search)", f"Search error: {es_e}")), 503

    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unknown error in /identify_and_search: {e}")
        return jsonify(fallback_response("identify_and_search", str(e))), 503

# --- NEW ROUTE (Integrated from previous turn) ---
@app.route("/search_possible_fish", methods=["POST"])
def search_possible_fish():
    """
    Input: JSON {"image": "user-upload/filename.jpg"}
    Output: Full JSON from AI (contains top_candidates, scores, reasons)
    """
    try:
        # 1. Parse JSON Input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
            
        image_key = data.get("image", "")
        if not image_key:
            return jsonify({"error": "No 'image' key provided in JSON"}), 400

        app.logger.info(f"Processing image search for key: {image_key}")

        # 2. Fetch Image from IBM COS
        try:
            cos = get_cos_client()
            response = cos.get_object(Bucket=COS_BUCKET_NAME, Key=image_key)
            image_bytes = response['Body'].read()
            pic_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
        except Exception as cos_error:
            app.logger.error(f"COS Error: {cos_error}")
            return jsonify({"error": f"Failed to fetch image: {str(cos_error)}"}), 500

        # 3. Call AI for fish identification
        ai_result = identify_fish_candidates_anthropic(pic_base64)
          

        print("this is ai_result",ai_result)

        # เช็คว่า AI ตอบกลับมาจริงไหม
        if not ai_result:
            app.logger.error("AI returned None or failed to parse JSON")
            return jsonify({"error": "AI could not identify fish"}), 500

        # Return ทั้งก้อน (Full Object)
        return jsonify(ai_result), 200

    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unhandled Error: {str(e)}")
        return jsonify(fallback_response("search_possible_fish", str(e))), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
=======
from flask import Flask, request, jsonify
from watsonx_captioning import get_fish_description_from_watsonxai, get_json_generated_image_details, get_json_generated_image_details_gemini, get_json_generated_image_details_groq
from elasticsearch_query import ElasticsearchQuery
from embedding_service import EmbeddingService
from function import return_top_n_fish, return_top_n_fish_simple, return_fish_info
from generation import get_generated_response, get_generated_response_with_context
import os
import time
import json
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config
import logging
from groq import Groq
import base64
import traceback
from fish_services import get_watsonx_token, identify_fish_candidates, identify_fish_candidates_gemini2, identify_fish_candidates_groq
from anthropic_captioning import (
    get_anthropic_client,
    caption_image_anthropic,
    identify_fish_details_anthropic,
    identify_fish_candidates_anthropic,
)
from openrouter_captioning import is_fish_image_openrouter, locate_fish_bbox_qwen, crop_b64_to_fish, cap_image_b64
from deepseek_captioning import rerank_candidates_deepseek
from concurrent.futures import ThreadPoolExecutor
from google import genai


load_dotenv()
es_endpoint = os.environ["es_endpoint"]
es_username = os.environ["es_username"]
es_password = os.environ["es_password"]
index_name = 'fish_index_v4'
esq = ElasticsearchQuery(es_endpoint, es_username, es_password)
emb = EmbeddingService('watsonx')

# Build each provider client only when its key is configured, so the service can
# start with whatever subset of providers is available. A missing key would
# otherwise crash the whole app at import (and crash-loop the pod on OpenShift).
_gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=_gemini_key) if _gemini_key else None

_groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=_groq_key) if _groq_key else None

watsonx_api_key = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
ibm_cloud_iam_url = os.getenv("IAM_IBM_CLOUD_URL", None)
chat_url = os.getenv("IBM_WATSONX_AI_INFERENCE_URL", None)

# Active vision provider for the image endpoints. /changeModel cycles only
# through providers whose credentials are actually configured (so cycling never
# lands on an unusable provider). The endpoints branch on ACTIVE_PROVIDER.
PROVIDER_CYCLE = ["anthropic", "gemini", "groq", "watsonx"]


def available_providers():
    """Providers (in cycle order) whose credentials are configured."""
    configured = {
        "anthropic": get_anthropic_client() is not None,
        "gemini": client is not None,
        "groq": groq_client is not None,
        "watsonx": bool(watsonx_api_key),
    }
    return [p for p in PROVIDER_CYCLE if configured[p]]


# Default to the first available provider (Anthropic first when present so the
# pipeline runs end-to-end out of the box); fall back to "anthropic" if none.
_available = available_providers()
ACTIVE_PROVIDER = _available[0] if _available else "anthropic"

# /identify_and_search Pass-2 reranker. The reranker re-scores the ES shortlist
# against the caption TEXT (it never sees the image). RERANK_PROVIDER selects the
# backend from RERANK_FUNCS below; to drop in a better model later, add one entry
# to that map (a fn taking (caption, candidates) and returning the shared
# text-rerank shape) and point RERANK_PROVIDER at it — no other code changes. Set
# it to anything not in the map (e.g. "none") to skip rerank and serve ES order.
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "deepseek").lower()
RERANK_FUNCS = {
    "deepseek": rerank_candidates_deepseek,
}

# Open-set / out-of-corpus gate for /identify_and_search. The gallery is tiny
# (~91 species), so a fish that is not in the database still maps to its nearest
# neighbors and reranks confidently. We flag (don't drop) low-confidence matches so
# the caller can route the image to the open-ended /search_possible_fish pipeline.
#
# Signal: ES has strong top-N recall (the true species is usually among the first
# few candidates even when it's not #1), so we trust the best per-candidate
# rerank match_score across the top IDENTIFY_TOP_N results rather than only the
# reranker's single best pick. On real out-of-corpus data, top match_score >= 0.40
# flags ~73% of unknowns while keeping ~73% of real in-DB fish confident (~27%
# false-flagged) -- the balanced operating point. Lower IDENTIFY_MATCH_MIN to 0.30 to
# rarely bother correct IDs (~47% caught / ~9% false-flag); raise it to catch more
# unknowns. NOTE: the gate is a MAX over the top-N, and the reranker orders results
# best-first, so IDENTIFY_TOP_N has little effect in practice -- the threshold is the
# real lever.
IDENTIFY_TOP_N = int(os.getenv("IDENTIFY_TOP_N", "5"))
IDENTIFY_MATCH_MIN = float(os.getenv("IDENTIFY_MATCH_MIN", "0.40"))
# Opt-in: when not confident, internally call the open-ended vision identifier and
# attach its output as open_search_results. Off by default (keeps latency/cost flat).
IDENTIFY_AUTO_FALLBACK = os.getenv("IDENTIFY_AUTO_FALLBACK", "0") == "1"
# Where to append the caption of each flagged out-of-corpus fish (JSONL, append-only)
# for later review / DB seeding. Logging is best-effort and never breaks the response.
UNKNOWN_FISH_LOG = os.getenv("UNKNOWN_FISH_LOG", os.path.join(os.path.dirname(__file__), "unknown_fish_captions.jsonl"))
_unknown_log_lock = threading.Lock()


def log_unknown_fish(entry):
    """Append one flagged-unknown record as a JSON line. Best-effort: any failure is
    logged and swallowed so a disk/permission issue can never fail the request."""
    try:
        line = json.dumps(entry, ensure_ascii=False)
        with _unknown_log_lock, open(UNKNOWN_FISH_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as log_e:
        app.logger.warning(f"Could not log unknown fish caption: {log_e}")

# --- Identification provider dispatch ---------------------------------------
# The two identification endpoints (/image_identification, /search_possible_fish)
# pick a provider by ACTIVE_PROVIDER. The underlying provider functions already
# return the same shape per operation, but their call signatures differ (some take
# a client, watsonx-candidates needs a token fetch first). These adapters normalize
# every provider to a uniform (b64) -> dict so the endpoints can dispatch through a
# single registry lookup. To add/swap a provider: write its function, wrap it here,
# add one map entry. An unknown ACTIVE_PROVIDER is a None lookup -> explicit error.


def _candidates_watsonx(b64):
    """WatsonX candidates adapter: fetch the IAM token, then identify."""
    token = get_watsonx_token(watsonx_api_key, ibm_cloud_iam_url)
    if not token:
        raise RuntimeError("WatsonX authentication failed: could not get access token")
    return identify_fish_candidates(b64, token, project_id, chat_url)


# Single best match -> {"image_contains_fish", "fish_details"}
IDENTIFY_DETAILS_FUNCS = {
    "anthropic": identify_fish_details_anthropic,
    "gemini": lambda b64: get_json_generated_image_details_gemini(client, b64),
    "groq": lambda b64: get_json_generated_image_details_groq(groq_client, b64),
    "watsonx": get_json_generated_image_details,
}

# Top-5 candidates -> {"image_contains_fish", "rejection_reason", "observed_features", "results"}
IDENTIFY_CANDIDATES_FUNCS = {
    "anthropic": identify_fish_candidates_anthropic,
    "gemini": lambda b64: identify_fish_candidates_gemini2(client, b64),
    "groq": lambda b64: identify_fish_candidates_groq(groq_client, b64),
    "watsonx": _candidates_watsonx,
}

app = Flask(__name__)

# COS bucket holding user-uploaded images. The image endpoints receive a COS
# object key (e.g. "user-upload/123.webp"), fetch the bytes here, then base64
# them for the vision models.
COS_BUCKET = os.getenv("IBM_COS_BUCKET", "fish-image-bucket")


def fetch_image_from_cos(image_key):
    """Fetch an object from COS by key and return its raw bytes.

    The image endpoints take a COS key (not inline base64); this centralizes the
    boto3 client construction + get_object so every endpoint loads images the
    same way. Raises on any COS error so callers can map it to a 503.
    """
    api_key = os.environ.get('IBM_COS_API_KEY')
    resource_instance_id = os.environ.get('IBM_COS_RESOURCE_INSTANCE_ID')
    endpoint_url = os.environ.get('IBM_COS_ENDPOINT')
    cos = ibm_boto3.client(
        's3',
        ibm_api_key_id=api_key,
        ibm_service_instance_id=resource_instance_id,
        config=Config(signature_version='oauth'),
        endpoint_url=endpoint_url
    )
    response = cos.get_object(Bucket=COS_BUCKET, Key=image_key)
    return response['Body'].read()

# Dummy fallback response
def fallback_response(service_name, error_msg=None):
    resp = {"error": f"{service_name} service unavailable", "fallback": True}
    if error_msg:
        resp["details"] = error_msg
    return resp

@app.route("/live", methods=["GET"])
def live():
    return jsonify(status="ok"), 200


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        text_input = data.get("text", "")
        if not text_input:
            return jsonify({"error": "No text input provided"}), 400

        caption_embedding = emb.embed_text(text_input)
        hits = esq.search_embedding(index_name=index_name, embedding_field='physical_description_embedding', query_vector=caption_embedding, size=5)
        top_n_fish = return_top_n_fish_simple(hits, n=5)
        return jsonify({"input": text_input, "results": top_n_fish})
    except Exception as e:
        print(f"Error in search: {e}")
        traceback.print_exc()
        app.logger.error(f"Error in search: {e}")
        return jsonify(fallback_response("search", f"error: {e} data {data}")), 503

# This service might take a while to respond due to image processing
@app.route("/image_captioning", methods=["POST"])
def image_captioning():
    try:
        data = request.get_json()
        image = data.get("image", "")
        app.logger.info(f"Received image: {image}")
        if not image:
            app.logger.error("No image provided in request")
            return jsonify({"error": "No image provided"}), 400

        # COS fetch + base64 block (image is a COS object key). Cap the long edge
        # before captioning (downscale-only) to trim upload bytes and billed pixels.
        try:
            app.logger.info(f"Fetching image from COS: {image}")
            image_bytes = fetch_image_from_cos(image)
            pic_string = cap_image_b64(base64.b64encode(image_bytes).decode('utf-8'))
        except Exception as cos_e:
            traceback.print_exc()
            app.logger.error(f"COS fetch error: {cos_e}")
            return jsonify(fallback_response("image_captioning", f"COS fetch error: {cos_e}")), 503

        # Vision model call block. Captioning is implemented for Anthropic and
        # WatsonX only (Gemini/Groq have no plain-caption function), so any
        # non-Anthropic provider falls back to WatsonX here.
        try:
            if ACTIVE_PROVIDER == "anthropic":
                app.logger.info("Calling Anthropic (Claude) for image captioning")
                caption = caption_image_anthropic(pic_string)
            else:
                app.logger.info("Calling WatsonX for image captioning")
                caption = get_fish_description_from_watsonxai(pic_string)
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"Captioning model error: {ai_e}")
            return jsonify(fallback_response("image_captioning", f"Captioning error: {ai_e}")), 503

        return jsonify({"caption": caption})
    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unknown error in image_captioning: {e}")
        return jsonify(fallback_response("image_captioning", str(e))), 503

@app.route("/image_identification", methods=["POST"])
def image_identification():
    try:
        data = request.get_json()
        image = data.get("image", "")
        app.logger.info(f"Received image: {image}")
        if not image:
            app.logger.error("No image provided in request")
            return jsonify({"error": "No image provided"}), 400

        # COS fetch + base64 block (image is a COS object key)
        try:
            app.logger.info(f"Fetching image from COS: {image}")
            image_bytes = fetch_image_from_cos(image)
            pic_string = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as cos_e:
            traceback.print_exc()
            app.logger.error(f"COS fetch error: {cos_e}")
            return jsonify(fallback_response("image_identification", f"COS fetch error: {cos_e}")), 503

        # Vision model call block
        try:
            identify_fn = IDENTIFY_DETAILS_FUNCS.get(ACTIVE_PROVIDER)
            if identify_fn is None:
                app.logger.error(f"Unknown ACTIVE_PROVIDER for identification: {ACTIVE_PROVIDER}")
                return jsonify(fallback_response("image_identification", f"Unknown provider: {ACTIVE_PROVIDER}")), 500
            app.logger.info(f"Calling {ACTIVE_PROVIDER} for image identification")
            json_result = identify_fn(pic_string)
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"Identification model error: {ai_e}")
            return jsonify(fallback_response("image_identification", f"Identification error: {ai_e}")), 503

        if not json_result:
            return jsonify({"error": "AI could not identify fish (Returned None)"}), 500

        return json_result
    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unknown error in image_identification: {e}")
        return jsonify(fallback_response("image_identification", str(e))), 503

@app.route("/generation", methods=["POST"])
def generation():
    try:
        data = request.get_json()
        question = data.get("question", "")
        chat_history = data.get("chat_history", [])  # List of previous messages
        context = data.get("context", "") #optioal context for the question

        if context:
            response_text = get_generated_response_with_context(question, context, chat_history)
        else:
            response_text = get_generated_response(question, chat_history)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error in generation: {e}")
        traceback.print_exc()
        app.logger.error(f"Error in generation: {e}")
        return jsonify(fallback_response("generation", str(e))), 503

@app.route("/search_with_scientific_name", methods=["POST"])
def search_with_scientific_name():
    try:
        data = request.get_json()
        scientific_name = data.get("scientific_name", "")
        if not scientific_name:
            return jsonify({"error": "No scientific name provided"}), 400

        print(f"Searching for scientific name: {scientific_name}")
        # Use text search on the scientific_name field, only 1 result
        hits = esq.search_text(index_name=index_name, field='scientific_name', text=scientific_name, size=1)

        fish_data = return_fish_info(hits)
        if not fish_data:
            return jsonify({
                "scientific_name": scientific_name,
                "fish_data": [],
                "message": "No fish found with the given scientific name."
            }), 200
        return jsonify({
            "scientific_name": scientific_name,
            "fish_data": fish_data,
            "message": "Success"
        }), 200
    except Exception as e:
        print(f"Error in search_with_scientific_name: {e}")
        traceback.print_exc()
        app.logger.error(f"Error in search_with_scientific_name: {e}")
        return jsonify({
            "scientific_name": scientific_name if 'scientific_name' in locals() else "",
            "fish_data": [],
            "message": f"Service error: {str(e)}"
        }), 503



# /identify_and_search is the primary fish-identification endpoint:
# caption -> Elasticsearch kNN -> LLM rerank (+ non-fish gate).
@app.route("/identify_and_search", methods=["POST"])
def identify_and_search():
    """
    1. Fetches the image from COS (image = COS object key).
    2. Runs the Qwen3-VL fish gate + crop localization in parallel; a non-fish
       verdict short-circuits, otherwise the crop tightens the image to caption.
    3. Generates a descriptive caption (Anthropic or WatsonX).
    4. Semantic vector search in Elasticsearch on the caption.
    5. DeepSeek text rerank re-scores the shortlist (ES order kept on failure).
    """
    try:
        data = request.get_json()
        image = data.get("image", "")
        if not image:
            app.logger.error("No image path provided in request for /identify_and_search")
            return jsonify({"error": "No image path (COS Key) provided"}), 400

        app.logger.info(f"Starting identify_and_search for image: {image}")

        # --- STEP 1: Fetch Image from COS and base64-encode ---
        try:
            app.logger.info("Loading COS credentials and fetching image.")
            image_bytes = fetch_image_from_cos(image)
            pic_string = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as cos_e:
            traceback.print_exc()
            app.logger.error(f"COS/Base64 error in identify_and_search: {cos_e}")
            return jsonify(fallback_response("identify_and_search (Image Load)", f"Image load error: {cos_e}")), 503

        # Per-stage timing (printed before the final response) to locate latency.
        _t0 = time.perf_counter()
        _t_gate = _t_caption = _t_es = _t0

        # --- STEP 1b: fish gate + crop localization, run IN PARALLEL ---
        # Both are independent Qwen3-VL calls on the same image, so we fire them
        # together: total latency is max(gate, crop) (~2s) instead of the sum
        # (~3.5s). The GATE is authoritative for fish/not-fish (the DeepSeek rerank
        # is text-only and can't reject non-fish photos); the CROP only tightens the
        # image we caption so the fish fills more of Sonnet's fixed pixel budget.
        # On a non-fish verdict we discard the crop and short-circuit, never paying
        # for the Sonnet caption / search / rerank. Both run for EVERY provider
        # (standalone OpenRouter calls), so swapping ACTIVE_PROVIDER never disables them.
        # To swap either model, set OPENROUTER_GATE_MODEL / OPENROUTER_CROP_MODEL.
        gate_contains_fish = None  # gate verdict, threaded into the final response
        caption_input = pic_string  # full image unless cropping succeeds
        pool = ThreadPoolExecutor(max_workers=2)
        gate_future = pool.submit(is_fish_image_openrouter, pic_string)
        crop_future = pool.submit(locate_fish_bbox_qwen, pic_string)

        rejection = None
        try:
            gate = gate_future.result()
            if isinstance(gate, dict):
                gate_contains_fish = gate.get("image_contains_fish")
            if gate_contains_fish is False:
                rejection = gate.get("rejection_reason")
        except Exception as gate_e:
            # Gate failure must not block a valid fish; log and continue to captioning.
            app.logger.warning(f"Fish gate check failed, continuing: {gate_e}")

        if rejection is not None:
            pool.shutdown(wait=False)  # don't block the response on the now-useless crop
            app.logger.info(f"Image rejected by fish gate: {rejection}")
            return jsonify({
                "ai_generated_caption": None,
                "observed_features": None,
                "elasticsearch_results": None,
                "reranked_results": None,
                "rerank_error": None,
                "image_contains_fish": False,
                "rejection_reason": rejection,
            })

        # Gate passed (or failed open): use the crop to tighten the captioned image.
        # crop_b64_to_fish returns the full image unchanged on a null/tiny box, so
        # cropping can only ever help — a localization miss never drops the fish.
        try:
            bbox = crop_future.result()
            caption_input = crop_b64_to_fish(pic_string, bbox)
            if caption_input != pic_string:
                app.logger.info("Captioning the cropped fish region.")
        except Exception as crop_e:
            app.logger.warning(f"Fish crop failed, using full image: {crop_e}")
        pool.shutdown(wait=False)
        # Cap the long edge before captioning. The vision model downscales larger
        # images anyway, so this trims upload bytes (and Sonnet's billed pixels)
        # with negligible detail loss. Downscale-only: no-op if already within cap.
        caption_input = cap_image_b64(caption_input)
        _t_gate = time.perf_counter()

        # --- STEP 2: Image Captioning (Pass 1) ---
        try:
            if ACTIVE_PROVIDER == "anthropic":
                app.logger.info("Calling Anthropic (Claude) for image captioning (Pass 1)")
                caption = caption_image_anthropic(caption_input)
            else:
                app.logger.info("Calling WatsonX for image captioning (Pass 1)")
                caption = get_fish_description_from_watsonxai(caption_input)
            app.logger.info(f"Generated Caption: {caption[:100]}...")
            if not caption:
                return jsonify({"error": "AI failed to generate a caption"}), 500
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"Captioning error in identify_and_search: {ai_e}")
            return jsonify(fallback_response("identify_and_search (Captioning)", f"AI error: {ai_e}")), 503
        _t_caption = time.perf_counter()

        # --- STEP 3: Semantic Search using Caption (Code copied from /search) ---
        try:
            app.logger.info("Starting Elasticsearch vector search with the generated caption.")
            text_input = caption
            caption_embedding = emb.embed_text(text_input)
            # Retrieve a wider shortlist (10) than we surface so the rerank pass has more
            # recall to work with — pure kNN often ranks the true species 6th-10th.
            hits = esq.search_embedding(index_name=index_name, embedding_field='physical_description_embedding', query_vector=caption_embedding, size=10)
            # Full fields (incl. physical_description) so the rerank pass has something to compare against.
            top_n_fish = return_top_n_fish(hits, n=10)
        except Exception as es_e:
            traceback.print_exc()
            app.logger.error(f"Elasticsearch search error in identify_and_search: {es_e}")
            return jsonify(fallback_response("identify_and_search (Elasticsearch Search)", f"Search error: {es_e}")), 503
        _t_es = time.perf_counter()

        # --- STEP 4: LLM Rerank (Pass 2) ---
        # Hand the ORIGINAL image + the ES shortlist back to the vision model and let it
        # re-score the candidates by actually looking at the photo. Pure kNN bunches its
        # scores and often mis-orders the shortlist; this lifts the true species to #1.
        reranked = None
        observed_features = None
        rerank_error = None
        image_contains_fish = gate_contains_fish
        rejection_reason = None
        # Rerank runs whenever the caption provider actually exists (a caption was
        # produced to rerank) — it is NOT tied to Anthropic. The reranker is chosen by
        # RERANK_PROVIDER (see RERANK_FUNCS); on any failure, or when RERANK_PROVIDER
        # names no known backend, we leave the ES ordering as-is rather than fall back
        # to another model. If no caption provider is configured there's nothing to rerank.
        rerank_fn = RERANK_FUNCS.get(RERANK_PROVIDER)
        if ACTIVE_PROVIDER in available_providers() and rerank_fn is not None:
            try:
                app.logger.info(f"Reranking ES shortlist with {RERANK_PROVIDER} (text) Pass 2.")
                rr = rerank_fn(caption, top_n_fish)
            except Exception as rr_e:
                # Reranker failed: keep the Elasticsearch results, no rerank.
                traceback.print_exc()
                app.logger.warning(f"{RERANK_PROVIDER} rerank failed, returning Elasticsearch results: {rr_e}")
                rerank_error = f"{RERANK_PROVIDER}: {rr_e}"
                rr = None
            if isinstance(rr, dict):
                reranked = rr.get("results")
                observed_features = rr.get("observed_features")
                # Only let the rerank override the gate verdict when it actually has one
                # (the text rerank returns None and should keep the gate's answer).
                if rr.get("image_contains_fish") is not None:
                    image_contains_fish = rr.get("image_contains_fish")
                    rejection_reason = rr.get("rejection_reason")
        else:
            app.logger.info(f"Rerank skipped: no caption provider configured or unknown RERANK_PROVIDER '{RERANK_PROVIDER}'.")
        _t_rerank = time.perf_counter()

        # --- STEP 5: Open-set verdict (is this fish actually in the database?) ---
        # ES has strong top-N recall, so we trust the BEST per-candidate rerank
        # match_score across the top IDENTIFY_TOP_N results: confident if it clears
        # IDENTIFY_MATCH_MIN. We FLAG but never drop the ranked list; the caller routes
        # unknowns to /search_possible_fish.
        top_scores = [r.get("rerank_score") for r in (reranked or [])[:IDENTIFY_TOP_N]
                      if r.get("rerank_score") is not None]
        top_match_score = max(top_scores) if top_scores else None
        match_confidence = top_match_score  # the value that drives the decision

        # Fail open: if the reranker produced no scores at all (e.g. an outage), don't
        # force a false unknown -- treat it as confident and let the ranked list stand.
        have_signal = top_match_score is not None
        confident_match = bool((not have_signal) or top_match_score >= IDENTIFY_MATCH_MIN)
        possible_new_species = not confident_match
        suggested_endpoint = "/search_possible_fish" if possible_new_species else None

        # Persist the caption of flagged unknowns as JSONL for later review / DB seeding.
        if possible_new_species:
            log_unknown_fish({
                "ts": datetime.now(timezone.utc).isoformat(),
                "match_confidence": match_confidence,
                "ai_generated_caption": caption,
                "observed_features": observed_features,
                # Top guesses we rejected -- useful context when reviewing the unknown.
                "top_candidates": [
                    {"fish_name": r.get("fish_name"), "scientific_name": r.get("scientific_name"),
                     "rerank_score": r.get("rerank_score"), "score": r.get("score")}
                    for r in (reranked or [])[:IDENTIFY_TOP_N]
                ],
            })

        # Opt-in: self-serve the open-ended identifier so a single call returns guesses.
        # Provider-agnostic: dispatch through the same registry the /search_possible_fish
        # endpoint uses, so it follows ACTIVE_PROVIDER instead of hardcoding one model.
        open_search_results = None
        if possible_new_species and IDENTIFY_AUTO_FALLBACK:
            identify_fn = IDENTIFY_CANDIDATES_FUNCS.get(ACTIVE_PROVIDER)
            if identify_fn is not None:
                try:
                    app.logger.info("Low confidence: running open-ended /search_possible_fish identifier.")
                    open_search_results = identify_fn(caption_input)
                except Exception as of_e:
                    traceback.print_exc()
                    app.logger.warning(f"Open-set auto-fallback failed: {of_e}")

        print(
            "[identify_and_search TIMING] "
            f"gate={_t_gate - _t0:.2f}s caption={_t_caption - _t_gate:.2f}s "
            f"embed+es={_t_es - _t_caption:.2f}s rerank={_t_rerank - _t_es:.2f}s "
            f"total={_t_rerank - _t0:.2f}s",
            flush=True,
        )

        # --- FINAL RESPONSE ---
        return jsonify({
            "ai_generated_caption": caption,
            "observed_features": observed_features,
            "elasticsearch_results": top_n_fish,
            "reranked_results": reranked,
            "rerank_error": rerank_error,
            "image_contains_fish": image_contains_fish,
            "rejection_reason": rejection_reason,
            # Open-set verdict: flag (don't drop) likely out-of-corpus fish.
            "match_confidence": match_confidence,  # best rerank match_score over top-N (drives the verdict)
            "confident_match": confident_match,
            "possible_new_species": possible_new_species,
            "suggested_endpoint": suggested_endpoint,
            "open_search_results": open_search_results,
        })

    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unknown error in /identify_and_search: {e}")
        return jsonify(fallback_response("identify_and_search", str(e))), 503

# --- Single vision call: image -> Top-5 candidates (no ES / rerank) ---
@app.route("/search_possible_fish", methods=["POST"])
def search_possible_fish():
    """
    Input: JSON {"image": "user-upload/filename.jpg"}  (COS object key)
    Output: Full JSON from AI (contains top_candidates, scores, reasons)
    """
    try:
        # 1. Parse JSON Input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        image_key = data.get("image", "")
        if not image_key:
            return jsonify({"error": "No 'image' key provided in JSON"}), 400

        app.logger.info(f"Processing image search for key: {image_key}")

        # 2. Fetch Image from IBM COS and base64-encode
        try:
            image_bytes = fetch_image_from_cos(image_key)
            pic_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as cos_error:
            traceback.print_exc()
            app.logger.error(f"COS Error: {cos_error}")
            return jsonify({"error": f"Failed to fetch image: {str(cos_error)}"}), 500

        # 3. Call the vision model (active provider)
        identify_fn = IDENTIFY_CANDIDATES_FUNCS.get(ACTIVE_PROVIDER)
        if identify_fn is None:
            app.logger.error(f"Unknown ACTIVE_PROVIDER for identification: {ACTIVE_PROVIDER}")
            return jsonify({"error": f"Unknown provider: {ACTIVE_PROVIDER}"}), 500
        print(f"Using {ACTIVE_PROVIDER} model for identification")
        ai_result = identify_fn(pic_base64)

        if not ai_result:
            app.logger.error("AI returned None or failed to parse JSON")
            return jsonify({"error": "AI could not identify fish"}), 500

        return jsonify(ai_result), 200

    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Unhandled Error: {str(e)}")
        return jsonify(fallback_response("search_possible_fish", str(e))), 500

@app.route("/changeModel", methods=["GET"])
def change_model():
    """Advance the active vision provider to the next configured one.

    Cycles anthropic -> gemini -> groq -> watsonx -> (wrap), skipping any
    provider whose credentials aren't set so it never lands on an unusable one.
    """
    global ACTIVE_PROVIDER
    avail = available_providers()
    if not avail:
        return jsonify({"provider": ACTIVE_PROVIDER, "available": [],
                        "error": "no providers configured"}), 200
    if ACTIVE_PROVIDER in avail:
        ACTIVE_PROVIDER = avail[(avail.index(ACTIVE_PROVIDER) + 1) % len(avail)]
    else:
        ACTIVE_PROVIDER = avail[0]
    app.logger.info(f"ACTIVE_PROVIDER set to: {ACTIVE_PROVIDER}")
    return jsonify({"provider": ACTIVE_PROVIDER, "available": avail}), 200

# Read-only status: which provider is active and which are configured.
@app.route("/currentModel", methods=["GET"])
def current_model():
    return jsonify({"provider": ACTIVE_PROVIDER, "available": available_providers()}), 200
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
