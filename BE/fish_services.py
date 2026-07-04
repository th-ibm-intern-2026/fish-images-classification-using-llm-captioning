import base64
import json
import os
import re
import anthropic
import http.client
from typing import Dict, Any, Optional
from fish_constants import SYSTEM_CONTENT_SINGLE


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
            {"role": "system", "content": SYSTEM_CONTENT_SINGLE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the marine organism. Return JSON with Top 5 candidates."},
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


def identify_fish_candidates_anthropic(pic_string: str) -> Optional[Dict[str, Any]]:
    """
    Identifies marine organisms from a base64-encoded image using Anthropic Claude vision.
    Returns scientific names for any identifiable species (fish, octopus, jellyfish, etc.)
    """
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_CONTENT_SINGLE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/webp",
                                "data": pic_string,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image. Identify the marine organism (fish, cephalopod, crustacean, marine mammal, prehistoric species, etc.) and return JSON with Top 5 candidates using scientific names."
                        }
                    ],
                }
            ],
        )

        ai_response = message.content[0].text
        print("▼▼▼▼▼▼ ANTHROPIC RAW OUTPUT ▼▼▼▼▼▼")
        print(ai_response[:500])
        print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲")

        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        clean_json_str = json_match.group(0) if json_match else ai_response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json_str)

    except Exception as e:
        print(f"Anthropic API Request Error: {e}")
        return None
