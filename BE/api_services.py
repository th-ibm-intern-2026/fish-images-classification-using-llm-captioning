from flask import Flask, request, jsonify
from watsonx_captioning import convert_image_to_base64, get_fish_description_from_watsonxai, get_json_generated_image_details, get_json_generated_image_details_gemini, get_json_generated_image_details_groq
from elasticsearch_query import ElasticsearchQuery
from embedding_service import EmbeddingService
from function import return_top_n_fish, return_top_n_fish_simple, return_fish_info
from generation import get_generated_response, get_generated_response_with_context
import os
import time
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config
import io
import logging
from groq import Groq
import base64
import traceback
from fish_services import get_watsonx_token, identify_fish_candidates, identify_fish_candidates_gemini, identify_fish_candidates_gemini2, identify_fish_candidates_gemini2, identify_fish_candidates_groq
from anthropic_captioning import (
    get_anthropic_client,
    caption_image_anthropic,
    identify_fish_details_anthropic,
    identify_fish_candidates_anthropic,
    rerank_candidates_haiku_text,
    is_fish_image_anthropic,
)
from deepseek_captioning import rerank_candidates_deepseek
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

# env
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

# /identify_and_search Pass-2 reranker. "deepseek" matches the caption TEXT against
# the shortlist (cheaper + higher top-1 in benchmarking) but never sees the image;
# "anthropic" re-sends the image to the vision model (also runs the non-fish gate).
# DeepSeek falls back to the Anthropic vision rerank if it errors or isn't configured.
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "deepseek").lower()

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

        # COS fetch + base64 block (image is a COS object key)
        try:
            app.logger.info(f"Fetching image from COS: {image}")
            image_bytes = fetch_image_from_cos(image)
            pic_string = base64.b64encode(image_bytes).decode('utf-8')
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
            if ACTIVE_PROVIDER == "anthropic":
              app.logger.info("Calling Anthropic (Claude) for image identification")
              json_result = identify_fish_details_anthropic(pic_string)
            elif ACTIVE_PROVIDER == "gemini":
              app.logger.info("Calling Gemini for image identification")
              json_result = get_json_generated_image_details_gemini(client, pic_string)
            elif ACTIVE_PROVIDER == "watsonx":
              app.logger.info("Calling WatsonX for image identification")
              json_result = get_json_generated_image_details(pic_string)
            else:
              app.logger.info("Calling Groq for image identification")
              json_result = get_json_generated_image_details_groq(groq_client, pic_string)
        except Exception as ai_e:
            traceback.print_exc()
            app.logger.error(f"Identification model error: {ai_e}")
            return jsonify(fallback_response("image_identification", f"Identification error: {ai_e}")), 503

        if not json_result:
            return jsonify({"error": "AI could not identify fish (Returned None)"}), 500

        print("this is json_result",json_result)
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



### Below this one
### /identify_and_search is the primary fish-identification endpoint:
### caption -> Elasticsearch kNN -> LLM rerank (+ non-fish gate).
@app.route("/identify_and_search", methods=["POST"])
def identify_and_search():
    """
    1. Fetches the image from COS (image = COS object key).
    2. (Anthropic only) cheap Haiku gate to reject non-fish photos up front.
    3. Generates a descriptive caption (Anthropic or WatsonX).
    4. Semantic vector search in Elasticsearch on the caption.
    5. LLM rerank (DeepSeek text, falling back to Haiku text) re-scores the shortlist.
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

        # --- STEP 1b: Cheap fish gate (Haiku) BEFORE the Sonnet caption ---
        # The DeepSeek rerank is text-only and can't reject non-fish photos, so we
        # screen the image up front with a small Haiku call. On a non-fish verdict we
        # short-circuit and never pay for the Sonnet caption / search / rerank.
        gate_contains_fish = None  # gate verdict, threaded into the final response
        if ACTIVE_PROVIDER == "anthropic":
            try:
                gate = is_fish_image_anthropic(pic_string)
                if isinstance(gate, dict):
                    gate_contains_fish = gate.get("image_contains_fish")
                if gate_contains_fish is False:
                    app.logger.info(f"Image rejected by fish gate: {gate.get('rejection_reason')}")
                    return jsonify({
                        "ai_generated_caption": None,
                        "observed_features": None,
                        "elasticsearch_results": None,
                        "reranked_results": None,
                        "rerank_error": None,
                        "image_contains_fish": False,
                        "rejection_reason": gate.get("rejection_reason"),
                    })
            except Exception as gate_e:
                # Gate failure must not block a valid fish; log and continue to captioning.
                app.logger.warning(f"Fish gate check failed, continuing: {gate_e}")
        _t_gate = time.perf_counter()

        # --- STEP 2: Image Captioning (Pass 1) ---
        try:
            if ACTIVE_PROVIDER == "anthropic":
                app.logger.info("Calling Anthropic (Claude) for image captioning (Pass 1)")
                caption = caption_image_anthropic(pic_string)
            else:
                app.logger.info("Calling WatsonX for image captioning (Pass 1)")
                caption = get_fish_description_from_watsonxai(pic_string)
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
        # Seed with the Haiku gate verdict (True once we got past the gate). The vision
        # rerank path can still override it below; the text rerank path keeps the gate's
        # answer since it never sees the image. None means "no verdict available".
        image_contains_fish = gate_contains_fish
        rejection_reason = None
        if ACTIVE_PROVIDER == "anthropic":
            rr = None
            # Preferred path: DeepSeek text rerank (caption vs. shortlist, no image).
            if RERANK_PROVIDER == "deepseek":
                try:
                    app.logger.info("Reranking ES shortlist with DeepSeek (text) Pass 2.")
                    rr = rerank_candidates_deepseek(caption, top_n_fish)
                except Exception as ds_e:
                    traceback.print_exc()
                    app.logger.warning(f"DeepSeek rerank failed, falling back to Anthropic vision: {ds_e}")
                    rerank_error = f"deepseek: {ds_e}"
            # Fallback path: Haiku TEXT rerank (same caption-vs-shortlist logic, no image).
            if rr is None:
                try:
                    app.logger.info("Reranking ES shortlist with Haiku (text) Pass 2.")
                    rr = rerank_candidates_haiku_text(caption, top_n_fish)
                    rerank_error = None  # Haiku text rerank recovered after a DeepSeek miss
                except Exception as rr_e:
                    traceback.print_exc()
                    app.logger.error(f"Rerank error in identify_and_search: {rr_e}")
                    rerank_error = str(rr_e)
            if isinstance(rr, dict):
                reranked = rr.get("results")
                observed_features = rr.get("observed_features")
                # Only let the rerank override the gate verdict when it actually has one
                # (the text rerank returns None and should keep the gate's answer).
                if rr.get("image_contains_fish") is not None:
                    image_contains_fish = rr.get("image_contains_fish")
                    rejection_reason = rr.get("rejection_reason")
        else:
            app.logger.info("Rerank skipped: active provider is not anthropic.")
        _t_rerank = time.perf_counter()

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
        ai_result = None

        if ACTIVE_PROVIDER == "anthropic":
          print("Using Anthropic (Claude) model for identification")
          ai_result = identify_fish_candidates_anthropic(pic_base64)
        elif ACTIVE_PROVIDER == "gemini":
          print("Using Gemini model for identification")
          ai_result = identify_fish_candidates_gemini2(client, pic_base64)
        elif ACTIVE_PROVIDER == "watsonx":
          print("Using WatsonX model for identification")
          access_token = get_watsonx_token(watsonx_api_key, ibm_cloud_iam_url)
          if not access_token:
            app.logger.error("Authentication failed: Could not get WatsonX token")
            return jsonify({"error": "Authentication failed"}), 500
          ai_result = identify_fish_candidates(pic_base64, access_token, project_id, chat_url)
        else:
          # Groq needs no WatsonX token.
          ai_result = identify_fish_candidates_groq(groq_client, pic_base64)


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
