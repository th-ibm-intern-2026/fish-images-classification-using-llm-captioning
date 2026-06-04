from flask import Flask, request, jsonify
from watsonx_captioning import convert_image_to_base64, get_fish_description_from_watsonxai, get_json_generated_image_details, get_json_generated_image_details_gemini, get_json_generated_image_details_groq
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
from groq import Groq
import base64
import traceback
from fish_services import get_watsonx_token, identify_fish_candidates, identify_fish_candidates_gemini, identify_fish_candidates_gemini2, identify_fish_candidates_gemini2, identify_fish_candidates_groq
from google import genai


load_dotenv()
es_endpoint = os.environ["es_endpoint"]
es_username = os.environ["es_username"]
es_password = os.environ["es_password"]
index_name = 'fish_index_v4'    
esq = ElasticsearchQuery(es_endpoint, es_username, es_password)
emb = EmbeddingService('watsonx')

global USE_GEMINI
USE_GEMINI = False 

client = genai.Client(
  api_key=os.getenv("GEMINI_API_KEY")
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
            app.logger.info("loading COS credentials")
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
            app.logger.info(f"Fetching image from COS: {image}")
            response = cos.get_object(Bucket='fish-image-bucket', Key=image)
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
            app.logger.info("loading COS credentials")
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
            app.logger.info(f"Fetching image from COS: {image}")
            response = cos.get_object(Bucket='fish-image-bucket', Key=image)
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

        # WatsonX call block
        try:
            # json_result = get_json_generated_image_details(pic_string)
            if USE_GEMINI:
              app.logger.info("Calling Gemini for image captioning")
              json_result = get_json_generated_image_details_gemini(client, pic_string)
            else:
              app.logger.info("Calling WatsonX for image captioning")
              json_result = get_json_generated_image_details_groq(groq_client, pic_string)
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

        # --- STEP 1: Fetch Image and Encode (Code copied from /image_captioning) ---
        try:
            app.logger.info("Loading COS credentials and fetching image.")
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
            response = cos.get_object(Bucket='fish-image-bucket', Key=image)
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
            api_key = os.environ.get('IBM_COS_API_KEY')
            resource_instance_id = os.environ.get('IBM_COS_RESOURCE_INSTANCE_ID')
            endpoint_url = os.environ.get('IBM_COS_ENDPOINT')
            bucket_name = 'fish-image-bucket' 

            cos = ibm_boto3.client(
                's3',
                ibm_api_key_id=api_key,
                ibm_service_instance_id=resource_instance_id,
                config=Config(signature_version='oauth'),
                endpoint_url=endpoint_url
            )
            
            response = cos.get_object(Bucket=bucket_name, Key=image_key)
            image_bytes = response['Body'].read()
            pic_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
        except Exception as cos_error:
            app.logger.error(f"COS Error: {cos_error}")
            return jsonify({"error": f"Failed to fetch image: {str(cos_error)}"}), 500

        # 3. Get Token & Call AI (Using fish_service)
        access_token = get_watsonx_token(watsonx_api_key, ibm_cloud_iam_url)
        if not access_token:
            app.logger.error("Authentication failed: Could not get WatsonX token")
            return jsonify({"error": "Authentication failed"}), 500

        # เรียก AI
        ai_result = None

        if USE_GEMINI:
          print("Using Gemini model for identification")
          ai_result = identify_fish_candidates_gemini2(client, pic_base64)
        else:
          # ai_result = identify_fish_candidates(pic_base64, access_token, project_id, chat_url)
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
def change_use_gemini():
    global USE_GEMINI
    USE_GEMINI = not USE_GEMINI
    app.logger.info(f"USE_GEMINI set to: {USE_GEMINI}")
    return jsonify({"USE_GEMINI": USE_GEMINI}), 200

@app.route("/isGemini", methods=["GET"])
def is_gemini():
    global USE_GEMINI
    return jsonify({"USE_GEMINI": USE_GEMINI}), 200
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
