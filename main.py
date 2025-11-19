# /my-python-ai-service/main.py
import os
import json
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGEN_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict"
VEO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

client = genai.Client(api_key=GEMINI_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:    Shared HTTP Client starting...")
    
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    yield
    
    await app.state.http_client.aclose()
    print("INFO:    Shared HTTP Client closed.")

app = FastAPI(lifespan=lifespan)

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

class DeepResearchRequest(BaseModel):
    query: str

async def generate_image_python(prompt: str, count: int, http_client: httpx.AsyncClient):
    request_body = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": count}
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    response = await http_client.post(IMAGEN_ENDPOINT, json=request_body, headers=headers)
    response.raise_for_status()
    
    gemini_response = response.json()
    predictions = gemini_response.get("predictions", [])
    
    if not predictions: 
        raise Exception("AI로부터 유효한 이미지를 생성하지 못했습니다.")
        
    base64_images = [p["bytesBase64Encoded"] for p in predictions if p.get("bytesBase64Encoded")]
    return base64_images

@app.post("/generate-image")
async def handle_generate_image(request: ImageRequest, fastapi_req: Request):
    try:
        http_client = fastapi_req.app.state.http_client
        base64_images = await generate_image_python(request.prompt, request.count, http_client)
        return {"status": "success", "images": base64_images}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-filter")
async def handle_generate_filter(request: FilterRequest):
    prompt = f"""
    You are an expert MongoDB query filter generator.
    (Generate a valid JSON object for MongoDB 'find' operation based on the user query.)
    Respond ONLY with the valid JSON object.
    
    [Task]
    User: "{request.user_id}"
    Query: "{request.query}"
    Current Time: "{request.current_time}"
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        text = response.text
        if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text: text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/describe-media")
async def handle_describe_media(request: DescriptionRequest, fastapi_req: Request):
    try:
        http_client = fastapi_req.app.state.http_client
        
        file_resp = await http_client.get(request.url)
        file_resp.raise_for_status()
        file_data = file_resp.content

        file_part = types.Part.from_bytes(data=file_data, mime_type=request.mime_type)

        if request.mime_type.startswith('image/'):
            prompt = "이 이미지를 데이터베이스 검색 항목으로 사용할 수 있도록 간결하고 사실적으로 묘사해 줘. 한국어로 답변해 줘."
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt, file_part]
            )
            return {"description": response.text}

        elif request.mime_type.startswith('text/'):
            text_content = file_data.decode('utf-8', errors='ignore')[:4000]
            prompt = f"이 텍스트 파일({request.file_name}) 내용을 데이터베이스 검색용으로 요약해 줘:\n\n{text_content}"
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt
            )
            return {"description": f"[텍스트 파일: {request.file_name}]\n{response.text}"}
        else:
            return {"description": f"(분석 미지원 파일: {request.file_name})"}

    except Exception as e:
        print(f"Description Error: {e}")
        return {"description": f"(AI 분석 실패: {request.file_name})"}

@app.post("/generate-video")
async def handle_generate_video(request: VideoRequest, fastapi_req: Request):
    endpoint = f"{VEO_BASE_URL}/models/veo-3.0-generate-001:predictLongRunning"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"instances": [{"prompt": request.prompt}]}
    
    http_client = fastapi_req.app.state.http_client

    try:
        resp = await http_client.post(endpoint, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/check-operation/{operation_name:path}")
async def check_operation(operation_name: str, fastapi_req: Request):
    url = f"{VEO_BASE_URL}/{operation_name}"
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    
    http_client = fastapi_req.app.state.http_client

    resp = await http_client.get(url, headers=headers)
    return resp.json()

@app.post("/deep-research")
def handle_deep_research(request: DeepResearchRequest):
    prompt = f"""
    You are a professional 'Deep Research Agent'.
    Your goal is to conduct a comprehensive investigation on the user's query using Google Search.
    
    [User Query]
    {request.query}
    
    [Instructions]
    1. Plan: Establish a search strategy.
    2. Search & Analyze: Use Google Search to find facts.
    3. Report: Write a detailed report in Korean.
    
    [Output Format (Korean Markdown)]
    # 📑 심층 리서치 보고서: [주제]
    ## 1. 📋 조사 계획
    ## 2. 🔍 주요 발견
    ## 3. 💡 결론
    """

    try:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=["TEXT"]
            )
        )
        
        report_text = response.text
        
        sources_text = "\n\n## 📚 참고 자료 (Sources)\n"
        
        if (response.candidates and 
            response.candidates[0].grounding_metadata and 
            response.candidates[0].grounding_metadata.grounding_chunks):
            
            chunks = response.candidates[0].grounding_metadata.grounding_chunks
            seen_urls = set()
            
            for i, chunk in enumerate(chunks, 1):
                if chunk.web and chunk.web.uri:
                    title = chunk.web.title or "제목 없음"
                    url = chunk.web.uri
                    
                    if url not in seen_urls:
                        sources_text += f"{i}. [{title}]({url})\n"
                        seen_urls.add(url)
        else:
            sources_text += "(참고한 웹 소스가 없거나 표시할 수 없습니다.)"

        final_report = report_text + sources_text
        
        return {"status": "success", "report": final_report}

    except Exception as e:
        print(f"Deep Research Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"status": "Python AI Service is running!"}