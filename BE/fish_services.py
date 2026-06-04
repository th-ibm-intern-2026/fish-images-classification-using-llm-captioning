# fish_service.py
import base64
import json
import re
from groq import Groq
import requests
import http.client
from typing import Dict, Any, Optional
from fish_constants import SYSTEM_CONTENT_SINGLE, MODEL_ID
from google import genai
from google.genai import types

def get_watsonx_token(api_key: str, iam_url: str) -> Optional[str]:
    try:
        if not api_key or not iam_url:
            return None
        # Clean URL if it contains protocol
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
    if not chat_url or not project_id:
        print("Missing URL or Project ID")
        return None
    
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT_SINGLE},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Identify the fish. Return JSON with Top 5 candidates."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pic_string}"}}
                ]
            }
        ],
        "project_id": project_id,
        "model_id": MODEL_ID,
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
            
            # Helper to extract JSON from markdown or raw text
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            clean_json_str = json_match.group(0) if json_match else ai_response.replace("```json", "").replace("```", "").strip()
            clean_json_str = clean_json_str.replace('\xa0', ' ')
            
            return json.loads(clean_json_str)
        return None
    except Exception as e:
        print(f"AI Request Error: {e}")
        return None

def identify_fish_candidates_gemini(client: genai.Client, pic_string: str) -> Optional[Dict[str, Any]]:
    try:
        # Decode base64 เป็น bytes
        try:
            image_bytes = base64.b64decode(pic_string)
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return None

        # Config Gemini
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            system_instruction=SYSTEM_CONTENT_SINGLE, # ใช้ Prompt ตัวเดียวกัน
            max_output_tokens=4096
        )

        # Call API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                  data=image_bytes,
                  mime_type="image/jpeg"
                ),
                'Identify the fish in this image'
            ],
            config=config
        )
        
        if response.text:
            return json.loads(response.text)
        return None

    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

def identify_fish_candidates_gemini2(client: genai.Client, pic_string: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes a base64 encoded image to identify fish species using Gemini.
    Enforces strict JSON output via Schema.
    """
    
    # ---------------------------------------------------------
    # 1. Define Schemas (Strict Output Control)
    # ---------------------------------------------------------
    
    # Schema สำหรับปลาแต่ละตัวใน list 'results'
    candidate_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "fish_name": types.Schema(
                type=types.Type.STRING, 
                description="Must be exactly one from the allowed species list."
            ),
            "score": types.Schema(
                type=types.Type.NUMBER, 
                description="Confidence score between 0.0 and 1.0"
            ),
            "score_reason": types.Schema(
                type=types.Type.STRING, 
                description="Brief explanation of visual features matching the description."
            )
        },
        required=["fish_name", "score", "score_reason"]
    )

    # Schema หลักของ JSON Response
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
                nullable=True # อนุญาตให้เป็น null ได้
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

        # Config Gemini
        # หมายเหตุ: SYSTEM_CONTENT_SINGLE ต้องเป็น f-string ที่ render ค่าตัวแปรมาครบแล้ว
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=main_schema, # 👈 หัวใจสำคัญ: บังคับโครงสร้าง
            temperature=0.1,             # ต่ำเพื่อให้ AI แม่นยำเรื่องชื่อและข้อมูล
            system_instruction=SYSTEM_CONTENT_SINGLE, 
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
                # ย้ำ Prompt สั้นๆ อีกครั้งเพื่อให้ AI เริ่มทำงาน
                'Analyze the image. Return JSON according to the schema.'
            ],
            config=config
        )
        
        # Parse Response
        # เพราะเราใช้ schema + application/json จึงมั่นใจได้ว่า text เป็น json แน่นอน
        if response.text:
            try:
                # 1. แปลง String เป็น Python Dict
                parsed_json = json.loads(response.text)
                
                # 2. Print แบบจัดระเบียบ (Pretty Print)
                print("▼▼▼▼▼▼ GEMINI JSON OUTPUT ▼▼▼▼▼▼")
                print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
                print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲")
                
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
    Identifies fish from a base64 string using Groq's Llama 4. maverick Vision model.
    """
    try:
        # Define the model. Groq supports Llama 3.2 Vision models.
        # Options: "llama-3.2-11b-vision-preview" or "llama-3.2-90b-vision-preview"
        # model_id = "llama-3.2-11b-vision-preview"
        groq_model_id = "meta-llama/llama-4-maverick-17b-128e-instruct"

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    # Important: For JSON mode to work, the word "JSON" must appear in the system prompt
                    "content": "You are a fish identification expert. Output strictly in JSON format. " 
                               + SYSTEM_CONTENT_SINGLE  # Replace with your actual variable
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Identify the fish. Return JSON with Top 5 candidates."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # Groq accepts data URLs for base64 images
                                "url": f"data:image/jpeg;base64,{pic_string}"
                            },
                        },
                    ],
                },
            ],
            model=groq_model_id,
            temperature=0,
            max_tokens=4096,
            top_p=1,
            stream=False,
            # This forces the model to return valid JSON, removing the need for regex cleaning
            response_format={"type": "json_object"}, 
        )

        # Extract content
        ai_response = chat_completion.choices[0].message.content
        
        # Parse JSON directly
        return json.loads(ai_response)

    except Exception as e:
        print(f"Groq API Request Error: {e}")
        return None