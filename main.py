# /my-python-ai-service/main.py

import os
import json
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGEN_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict"
VEO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Gemini SDK 설정
genai.configure(api_key=GEMINI_API_KEY)
flash_model = genai.GenerativeModel('gemini-2.5-flash')
pro_model = genai.GenerativeModel('gemini-2.5-pro')

app = FastAPI()

# --- 데이터 모델 정의 (Pydantic) ---
class ImageRequest(BaseModel):
    prompt: str
    count: int = 1

class FilterRequest(BaseModel):
    query: str
    user_id: str
    current_time: str

class DescriptionRequest(BaseModel):
    url: str
    mime_type: str
    file_name: str

class VideoRequest(BaseModel):
    prompt: str

async def generate_image_python(prompt: str, count: int = 1):
    request_body = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": count}
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    # httpx를 사용해 비동기로 Google API 호출
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                IMAGEN_ENDPOINT,
                json=request_body,
                headers=headers,
                timeout=120.0 # 타임아웃 120초
            )
            response.raise_for_status() # 200~300번대가 아니면 에러 발생
            
            gemini_response = response.json()
            predictions = gemini_response.get("predictions", [])
            
            if not predictions:
                raise Exception("AI로부터 유효한 이미지를 생성하지 못했습니다.")

            # Base64 인코딩된 이미지 문자열 리스트를 반환
            base64_images = [p["bytesBase64Encoded"] for p in predictions if p.get("bytesBase64Encoded")]
            return base64_images

        except httpx.HTTPStatusError as e:
            # Google API에서 에러 응답이 온 경우
            print(f"Gemini Imagen API Error: {e.response.text}")
            raise Exception(f"AI 이미지 생성 오류: {e.response.status_code}")
        except Exception as e:
            print(f"generate_image_python 예외 발생: {e}")
            raise e

# --- 1. 이미지 생성 (기존 코드 유지) ---
@app.post("/generate-image")
async def handle_generate_image(request: ImageRequest):
    try:
        base64_images = await generate_image_python(request.prompt, request.count)
        
        return {"status": "success", "images": base64_images}
    
    except Exception as e:
        return {"status": "error", "message": str(e)} 


# --- 2. MongoDB 필터 생성 (Gemini SDK 사용) ---
@app.post("/generate-filter")
async def handle_generate_filter(request: FilterRequest):
    prompt = f"""
    You are an expert MongoDB query filter generator.
    (중략... utils/ai_helper.js의 프롬프트 내용을 그대로 가져오거나 요약해서 넣음)
    Respond ONLY with the valid JSON object.
    
    [Task]
    User: "{request.user_id}"
    Query: "{request.query}"
    Current Time: "{request.current_time}"
    """
    
    try:
        response = flash_model.generate_content(prompt)
        text = response.text
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        return json.loads(text)
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- 3. 파일 설명/요약 (이미지/텍스트) ---
@app.post("/describe-media")
async def handle_describe_media(request: DescriptionRequest):
    try:
        # 1. Python 서버가 직접 파일 다운로드
        async with httpx.AsyncClient() as client:
            file_resp = await client.get(request.url)
            file_resp.raise_for_status()
            file_data = file_resp.content

        # 2. 미디어 타입에 따른 처리
        if request.mime_type.startswith('image/'):
            prompt = "이 이미지를 데이터베이스 검색 항목으로 사용할 수 있도록 간결하고 사실적으로 묘사해 줘. 한국어로 답변해 줘."
            response = pro_model.generate_content([
                prompt,
                {'mime_type': request.mime_type, 'data': file_data}
            ])
            return {"description": response.text}

        elif request.mime_type.startswith('text/'):
            text_content = file_data.decode('utf-8', errors='ignore')[:4000]
            prompt = f"이 텍스트 파일({request.file_name}) 내용을 데이터베이스 검색용으로 요약해 줘:\n\n{text_content}"
            response = flash_model.generate_content(prompt)
            return {"description": f"[텍스트 파일: {request.file_name}]\n{response.text}"}

        else:
            return {"description": f"(분석 미지원 파일: {request.file_name})"}

    except Exception as e:
        print(f"Description Error: {e}")
        return {"description": f"(AI 분석 실패: {request.file_name})"}


# --- 4. 비디오 생성 (Veo, REST API) ---
@app.post("/generate-video")
async def handle_generate_video(request: VideoRequest):
    endpoint = f"{VEO_BASE_URL}/models/veo-3.0-generate-001:predictLongRunning"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    body = {"instances": [{"prompt": request.prompt}]}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

# 상태 확인용 엔드포인트
@app.get("/check-operation/{operation_name:path}")
async def check_operation(operation_name: str):
    url = f"{VEO_BASE_URL}/{operation_name}"
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        return resp.json()