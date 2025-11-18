# /my-python-ai-service/main.py

import os
import httpx # Node.js의 fetch/axios 같은 역할
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드

# 1. 환경 변수 및 FastAPI 앱 초기화
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGEN_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict"

app = FastAPI()

# 2. Node.js가 요청할 때 받을 JSON 형태 정의
class ImageRequest(BaseModel):
    prompt: str
    count: int = 1

# 3. ai_helper.js의 generateImage 함수를 파이썬으로 구현
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

# 4. API 엔드포인트 생성
@app.post("/generate-image")
async def handle_generate_image(request: ImageRequest):
    try:
        # 파이썬 로직 호출
        base64_images = await generate_image_python(request.prompt, request.count)
        
        # 성공 시, base64 문자열이 담긴 JSON 반환
        return {"status": "success", "images": base64_images}
    
    except Exception as e:
        # 실패 시, 에러 메시지 반환
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"status": "Python AI Service is running!"}