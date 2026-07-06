import base64
import json
import os
import re
import anthropic
import http.client
from groq import Groq
from typing import Dict, Any, Optional

from fish_constants import SYSTEM_CONTENT_OPEN, MODEL_ID
from google import genai
from google.genai import types

# Shared instruction for every provider: make an open-ended guess (not limited to any
# list) and return the Top 5 most likely species with the visible features behind each.
IDENTIFY_USER_PROMPT = (
    "Identify the fish species in the image. You are not limited to any list — give your "
    "best open-ended guess. Return JSON with your Top 5 most likely species, each with a "
    "confidence score and the visible features behind the guess."
)

def get_watsonx_token(api_key: str, iam_url: str) -> Optional[str]:
    try:
        if not api_key or not iam_url:
            return None
        host = iam_url.replace("https://", "").replace("http://", "").rstrip('/')
        conn = http.client.HTTPSConnection(host)
        payload = f"grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey&apikey={api_key}"
        headers = {'Content-Type': "application/x-www-form-urlencoded"}
        conn.request("POST", "/identity/token", payload, headers)
        res = conn.getresponse()
        data = res.read()
        if res.status == 200:
            return json.loads(data.decode("utf-8")).get("access_token")
        print(f"Token Error: {res.status} - {data}")
        return None
    except Exception as e:
        print(f"Error getting token: {e}")
        return None


def identify_fish_candidates(pic_string: str, access_token: str, project_id: str, chat_url: str) -> Optional[Dict[str, Any]]:
    """WatsonX-based identification (legacy, kept for compatibility)."""
    import requests
    if not chat_url or not project_id:
        print("Missing URL or Project ID")
        return None

    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT_OPEN},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the marine organism. Return JSON with Top 5 candidates."},
                    {"type": "text", "text": IDENTIFY_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pic_string}"}}
                ]
            }
        ],
        "project_id": project_id,
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "decoding_method": "greedy",
        "repetition_penalty": 1.1,
        "max_tokens": 4096,
        "temperature": 0
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    try:
        response = requests.post(chat_url, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            ai_response = data['choices'][0]['message']['content']
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            clean_json_str = json_match.group(0) if json_match else ai_response.replace("```json", "").replace("```", "").strip()
            clean_json_str = clean_json_str.replace('\xa0', ' ')
            return json.loads(clean_json_str)
        return None
    except Exception as e:
        print(f"AI Request Error: {e}")
        return None


def identify_fish_candidates_gemini2(client: genai.Client, pic_string: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes a base64 encoded image to identify fish species using Gemini.
    Enforces strict JSON output via Schema.
    """
    
    # ---------------------------------------------------------
    # 1. Define Schemas (Strict Output Control)
    # ---------------------------------------------------------
    
    # Schema for each candidate in the 'results' list
    candidate_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "fish_name": types.Schema(
                type=types.Type.STRING,
                description="Common name of the identified species (open-ended, not from any list)."
            ),
            "score": types.Schema(
                type=types.Type.NUMBER,
                description="Confidence score between 0.0 and 1.0"
            ),
            "score_reason": types.Schema(
                type=types.Type.STRING,
                description="The visible features that led to this guess."
            )
        },
        required=["fish_name", "score", "score_reason"]
    )

    # Top-level JSON response schema
    main_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "image_contains_fish": types.Schema(
                type=types.Type.BOOLEAN,
                description="True only if valid, raw/fresh fish is detected."
            ),
            "rejection_reason": types.Schema(
                type=types.Type.STRING,
                description="Reason if image_contains_fish is false, otherwise null.",
                nullable=True
            ),
            "results": types.Schema(
                type=types.Type.ARRAY,
                items=candidate_schema,
                description="List of top 5 candidates. Empty if image_contains_fish is false."
            )
        },
        required=["image_contains_fish", "results"] # rejection_reason is optional in JSON if null
    )

    # ---------------------------------------------------------
    # 2. Logic & Execution
    # ---------------------------------------------------------
    try:
        # Decode Image
        try:
            image_bytes = base64.b64decode(pic_string)
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return None

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=main_schema,  # force the output structure
            temperature=0.1,              # low temperature for accurate names/info
            system_instruction=SYSTEM_CONTENT_OPEN,
            max_output_tokens=4096
        )

        # Call API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/webp"
                ),
                IDENTIFY_USER_PROMPT
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

    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

def identify_fish_candidates_groq(client: Groq, pic_string: str) -> Optional[Dict[str, Any]]:

    """
    Identifies marine organisms from a base64-encoded image using Groq's vision model.
    Returns the open-ended Top-5 candidates shape ({"image_contains_fish",
    "rejection_reason", "results"}) with SCIENTIFIC names, matching the other
    candidate providers.
    """
    # Groq Vision model (OpenAI-compatible chat.completions API).
    groq_model_id = "meta-llama/llama-4-maverick-17b-128e-instruct"

    try:
        completion = client.chat.completions.create(
            model=groq_model_id,
            messages=[
                {"role": "system", "content": SYSTEM_CONTENT_OPEN},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IDENTIFY_USER_PROMPT},
                        {
                            "type": "image_url",
                            # Groq accepts data URLs for base64 images.
                            "image_url": {"url": f"data:image/jpeg;base64,{pic_string}"},
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=4096,
            # JSON mode forces structured output (SYSTEM_CONTENT_OPEN already says "JSON").
            response_format={"type": "json_object"},
        )

        ai_response = completion.choices[0].message.content
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        clean_json_str = json_match.group(0) if json_match else ai_response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json_str)

    except Exception as e:
        print(f"Groq API Request Error: {e}")
        return None
