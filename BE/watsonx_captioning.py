from dotenv import load_dotenv
import base64
import os
import http.client
import json
from groq import Groq
from google import genai
from google.genai import types
import requests
from google.genai.errors import APIError
from typing import Optional, Dict, Any
from fish_constants import SYSTEM_CONTENT_DETAILS

load_dotenv()

watsonx_api_key = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
ibm_cloud_iam_url = os.getenv("IAM_IBM_CLOUD_URL", None)
chat_url = os.getenv("IBM_WATSONX_AI_INFERENCE_URL", None)


def get_fish_description_from_watsonxai(pic_string):
    conn_ibm_cloud_iam = http.client.HTTPSConnection(ibm_cloud_iam_url)
    payload = "grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey&apikey="+watsonx_api_key
    headers = { 'Content-Type': "application/x-www-form-urlencoded" }
    conn_ibm_cloud_iam.request("POST", "/identity/token", payload, headers)
    res = conn_ibm_cloud_iam.getresponse()
    data = res.read()
    decoded_json=json.loads(data.decode("utf-8"))
    access_token=decoded_json["access_token"]

    system_content = """
    You always answer the questions with markdown formatting using GitHub syntax. 
    The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. 
    You must omit that you answer the questions with markdown.

    Any HTML tags must be wrapped in block quotes, for example:
    ```<html>```. 
    You will be penalized for not rendering code in block quotes.

    When returning code blocks, specify the language.

    You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
    """
    user_message = """Please provide a detailed description of what the image depicts and what you think it"""

    body = {
    "messages": [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": user_message,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64, {pic_string}"
                }
                }
            ]
        }
    ],
    "project_id": project_id,
    "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "decoding_method": "greedy",
    "repetition_penalty": 1.1,
    "max_tokens": 900
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.post(
        chat_url,
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    return data['choices'][0]['message']['content']

def get_json_generated_image_details(pic_string):
    conn_ibm_cloud_iam = http.client.HTTPSConnection(ibm_cloud_iam_url)
    payload = "grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey&apikey="+watsonx_api_key
    headers = { 'Content-Type': "application/x-www-form-urlencoded" }
    conn_ibm_cloud_iam.request("POST", "/identity/token", payload, headers)
    res = conn_ibm_cloud_iam.getresponse()
    data = res.read()
    decoded_json=json.loads(data.decode("utf-8"))
    access_token=decoded_json["access_token"]

    system_content = SYSTEM_CONTENT_DETAILS

    user_message = """
    Analyze the provided image. Identify the species and return the detailed JSON object as defined in your system instructions.
    """

    body = {
    "messages": [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": user_message,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64, {pic_string}"
                }
                }
            ]
        }
    ],
    "project_id": project_id,
    "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "decoding_method": "greedy",
    "repetition_penalty": 1.1,
    "max_tokens": 900
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.post(
        chat_url,
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    json_string = data['choices'][0]['message']['content']

    # Validate and parse the JSON response
    try:
        json_data = json.loads(json_string)
        if not isinstance(json_data, dict):
            raise ValueError("Response is not a valid JSON object")
        if "image_contains_fish" not in json_data or "fish_details" not in json_data:
            raise ValueError("Response JSON is missing required keys")
        if json_data["image_contains_fish"] and not json_data["fish_details"]:
            raise ValueError("Fish details should not be empty if image_contains_fish is true")
        return json_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e} returned: {json_string}")

def get_json_generated_image_details_groq(groq_client: Groq ,pic_string: str):
    """
    Analyzes a base64 encoded image using Groq's Vision model (Llama 3.2 Vision)
    and returns a structured JSON response based on the defined schema.
    """

    SYSTEM_CONTENT_FULL = SYSTEM_CONTENT_DETAILS

    user_message = "Analyze the provided image. Identify the species and return the detailed JSON object as defined in your system instructions."

    # Groq Vision Model (replace with your preference: 11b or 90b)
    groq_model_id = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    try:
        completion = groq_client.chat.completions.create(
            model=groq_model_id,
            messages=[
                {"role": "system", "content": SYSTEM_CONTENT_FULL},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Groq accepts the data URL format
                                "url": f"data:image/jpeg;base64,{pic_string}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0, # Use low temperature for factual/structured tasks
            max_tokens=900,
            # Use JSON mode to force structured output, reducing parsing errors
            response_format={"type": "json_object"}, 
        )

        # Groq guarantees valid JSON in the response.content when JSON mode is used.
        json_string = completion.choices[0].message.content
        
        # 1. Parse JSON
        json_data = json.loads(json_string)

        # 2. Final validation (optional but recommended for robustness)
        if not isinstance(json_data, dict):
            raise ValueError("Response is not a valid JSON object (post-parse check)")
        if "image_contains_fish" not in json_data or "fish_details" not in json_data:
            raise ValueError("Response JSON is missing required keys")
        if json_data["image_contains_fish"] and not json_data["fish_details"]:
            # Note: This check relies on the model filling the details correctly
            print("Warning: image_contains_fish is true, but fish_details is empty.")

        return json_data

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e} - Response: {json_string}")
        return None
    except Exception as e:
        print(f"Groq API Request Error: {e}")
        return None
      
      
def get_json_generated_image_details_gemini(client: genai.Client, pic_string: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes a base64 encoded image using Gemini's Vision model
    and returns a structured JSON response based on the defined schema.
    """
    
    # ---------------------------------------------------------
    # 1. Define JSON Schemas (Defined locally inside function)
    # ---------------------------------------------------------
    
    # Schema for fish details
    fish_details_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "fish_name": types.Schema(type=types.Type.STRING, description="Common name in English."),
            "scientific_name": types.Schema(type=types.Type.STRING, description="Scientific Latin name."),
            "order_name": types.Schema(type=types.Type.STRING, description="Taxonomic Order in English."),
            "physical_description": types.Schema(type=types.Type.STRING, description="Comprehensive 3-5 sentence physical description IN THAI LANGUAGE."),
            "habitat": types.Schema(type=types.Type.STRING, description="Detailed habitat description IN THAI LANGUAGE (e.g., water type, depth, environment)."),
        },
        required=["fish_name", "scientific_name", "order_name", "physical_description", "habitat"]
    )

    # Top-level schema
    main_json_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "image_contains_fish": types.Schema(type=types.Type.BOOLEAN, description="True if the image contains a valid, fresh fish specimen."),
            "fish_details": fish_details_schema, 
        },
        required=["image_contains_fish", "fish_details"],
    )

    # ---------------------------------------------------------
    # 2. Define System Prompt
    # ---------------------------------------------------------
    system_content_full = SYSTEM_CONTENT_DETAILS

    user_message = "Analyze the provided image. Identify the species and return the detailed JSON object as defined in your system instructions."

    # ---------------------------------------------------------
    # 3. Execution Logic
    # ---------------------------------------------------------
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(pic_string)

        # Prepare image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/webp' 
        )

        # Config Gemini
        config = types.GenerateContentConfig(
            response_schema=main_json_schema,  # use the schema declared above
            response_mime_type="application/json",
            temperature=0.0, 
            system_instruction=system_content_full,
            max_output_tokens=900
        )

        # Call API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                image_part,
                user_message
            ],
            config=config
        )

        # Parse Response
        if response.text:
            try:
                parsed_json = json.loads(response.text)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Gemini: {e}")
                print(f"Raw text was: {response.text}")
                return None
        else:
            print("Gemini returned empty response (Check Safety Settings or Image Quality)")
            return None

    except APIError as e:
        print(f"Gemini API Error: {e}")
        return None
    except Exception as e:
        print(f"General Error in Gemini function: {e}")
        return None