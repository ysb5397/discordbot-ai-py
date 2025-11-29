# /my-python-ai-service/main.py
import os
import json
import httpx
import io
import base64
import asyncio
import platform
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import FinanceDataReader as fdr
import requests
import re
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from datetime import datetime, timedelta
from matplotlib.figure import Figure
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

IMAGEN_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-ultra-generate-001:predict"
VEO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

AI_PERSONA = os.getenv("AI_PERSONA", """
너는 사용자의 친한 친구이자 유능한 AI 비서야. 
설명은 친절하고 귀엽게 반말(해체)로 해줘. 
전문적인 내용이라도 쉽고 재미있게 풀어서 설명해줘.
""")

client = genai.Client(api_key=GEMINI_API_KEY)

def get_font_prop():
    font_name = get_font_family()
    # 폰트 경로를 찾거나 시스템 폰트 이름 사용
    return fm.FontProperties(family=font_name)

# --- 헬퍼 함수: 한글 폰트 설정 (차트용) ---
def get_font_family():
    system_name = platform.system()
    if system_name == 'Windows': return 'Malgun Gothic'
    elif system_name == 'Darwin': return 'AppleGothic'
    else: return 'NanumGothic' # Dockerfile에 폰트 설치 필요 (없으면 깨짐)

# --- 헬퍼 함수: 네이버 뉴스 검색 ---
def fetch_naver_news(keyword, display=10):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return []
    
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {"query": keyword, "display": display, "sort": "sim"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            items = resp.json().get('items', [])
            news_list = []
            for item in items:
                title = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', item['title'])
                news_list.append(f"- {title} ({item['pubDate'][:16]})")
            return news_list
    except Exception as e:
        print(f"Naver News Error: {e}")
    return []

# --- 헬퍼 함수: 차트 그리기 (Blocking I/O) ---
def draw_stock_chart(df, title):
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    
    font_prop = get_font_prop()

    ax.plot(df.index, df['Close'], label='Close', color='#333333')
    
    # 이동평균선 (데이터가 충분할 때만)
    if len(df) > 20:
        df['MA20'] = df['Close'].rolling(window=20).mean()
        ax.plot(df.index, df['MA20'], label='MA20', color='red', linestyle='--')
    if len(df) > 60:
        df['MA60'] = df['Close'].rolling(window=60).mean()
        ax.plot(df.index, df['MA60'], label='MA60', color='blue', linestyle='--')

    ax.set_title(f"{title} Stock Price", fontsize=15, fontproperties=font_prop)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(prop=font_prop)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 이미지를 메모리 버퍼에 저장
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=100)

    fig.clear() # Figure 객체 정리
    buf.seek(0)
    
    # Base64 인코딩
    return base64.b64encode(buf.getvalue()).decode('utf-8')

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
    aspectRatio: str = "1:1"
    resolution: str = "1K"
    referenceImageUrl: str | None = None
    mimeType: str | None = None

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

class CodeReviewRequest(BaseModel):
    diff: str

class StockAnalyzeRequest(BaseModel):
    query: str

async def generate_image_python(request: ImageRequest, http_client: httpx.AsyncClient):
    contents = [request.prompt]
    
    if request.referenceImageUrl:
        try:
            print(f"Downloading reference image: {request.referenceImageUrl}")
            img_resp = await http_client.get(request.referenceImageUrl)
            img_resp.raise_for_status()
            
            image_bytes = io.BytesIO(img_resp.content)
            pil_image = Image.open(image_bytes)
            contents.append(pil_image)
        except Exception as e:
            print(f"Failed to download/process reference image: {e}")

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=request.aspectRatio,
                    image_size=request.resolution
                ),
            )
        )

        base64_images = []
        
        if response.parts:
            for part in response.parts:
                if part.inline_data and part.inline_data.data:
                     raw_bytes = part.inline_data.data
                     b64_str = base64.b64encode(raw_bytes).decode('utf-8')
                     base64_images.append(b64_str)
                elif hasattr(part, 'as_image'):
                     try:
                         img = part.as_image()
                         buffered = io.BytesIO()
                         img.save(buffered, format="PNG")
                         b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                         base64_images.append(b64_str)
                     except Exception as img_err:
                         print(f"Image conversion error: {img_err}")

        if not base64_images:
            raise Exception("AI가 이미지를 생성하지 않았거나 응답 형식이 다릅니다.")

        return base64_images

    except Exception as e:
        print(f"GenAI Image Generation Error: {e}")
        raise e

@app.post("/generate-image")
async def handle_generate_image(request: ImageRequest, fastapi_req: Request):
    try:
        http_client = fastapi_req.app.state.http_client
        
        base64_images = await generate_image_python(request, http_client)
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
    # 꼼꼼한 분석가 페르소나 주입 (겉은 귀엽게, 속은 치밀하게)
    prompt = f"""
    {AI_PERSONA}
    
    하지만 이번 작업에서 너는 **'세계 최고의 심층 분석가'** 모드로 작동해야 해.
    사용자의 질문에 대해 대충 대답하지 말고, 집요하게 파고들어서 팩트를 검증해 줘.

    [사용자 요청]
    {request.query}

    [생각의 사슬 (Chain of Thought) - 이 순서를 반드시 지켜!]
    1. **Plan**: 무엇을 검색해야 완벽한 답을 얻을 수 있을지 전략을 세운다.
    2. **Search & Analyze**: Google Search를 통해 정보를 수집하고 분석한다.
    3. **Critique (비판적 재검토)**: 수집한 정보에 부족한 점은 없는지, 편향되지는 않았는지 스스로 반문하고 보완한다.
    4. **Drafting**: 수집된 정보를 바탕으로 방대한 양의 '상세 보고서'와 핵심만 요약한 '브리핑'을 작성한다.

    [최종 출력 형식 (Strict Output Format)]
    분석이 끝나면 반드시 아래 XML 태그 형식을 엄격하게 지켜서 답변해. 
    다른 잡담은 태그 밖에 쓰지 마.

    <REPORT_FILE>
    # 📑 심층 리서치 보고서: [주제]
    (여기에 마크다운(Markdown) 형식으로 아주 상세하게 작성해. 논문 수준으로 깊이 있게. 
    출처(Source) 링크도 꼼꼼하게 달아줘. 길이 제한 없이 마음껏 써도 돼.)
    </REPORT_FILE>

    <DISCORD_EMBED>
    (여기에 디스코드 채팅창에 보여줄 내용을 작성해.)
    - **분량**: 300자~500자 이내.
    - **말투**: 너의 원래 페르소나(귀여운 반말)를 유지해. 이모지는 방해되지 않을만큼 적당히 활용!
    - **내용**: 
      1. 조사를 통해 알아낸 가장 충격적이거나 중요한 3가지 포인트 (글머리 기호)
      2. 너의 한 줄 총평
      3. "자세한 내용은 위에 첨부한 파일 읽어봐! 📄" 라는 멘트로 마무리.
    </DISCORD_EMBED>
    """

    try:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        response = client.models.generate_content(
            model='gemini-2.5-pro', 
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=["TEXT"],
                temperature=0.4
            )
        )
        
        return {"status": "success", "report": response.text}

    except Exception as e:
        print(f"Deep Research Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/code-review")
def handle_code_review(request: CodeReviewRequest):
    prompt = f"""
    {AI_PERSONA}
    
    이번엔 **'Google 수석 엔지니어 겸 보안 전문가'** 모드야.
    아래 제공된 [Git Diff]는 지난 일주일 동안 변경된 서버 코드야. 
    이 변경 사항들을 아주 꼼꼼하게 점검해줘.

    [Git Diff (이번 주 변경 사항)]
    {request.diff}

    [생각의 사슬 (Chain of Thought)]
    1. **Scan**: 변경된 파일과 로직의 의도를 먼저 파악한다.
    2. **Deep Dive**: 
       - 🐛 버그 가능성: 엣지 케이스(Edge case) 처리 미흡, 타입 에러 등.
       - 🛡️ 보안 취약점: SQL Injection, XSS, 민감 정보 노출 등.
       - ⚡ 성능 이슈: 불필요한 루프, 메모리 누수, 비효율적인 DB 쿼리.
       - 🧹 가독성: 변수명, 함수 구조, 중복 코드.
    3. **Critique**: "이게 최선인가?" 스스로 반문하며 더 나은 대안(Best Practice)을 생각한다.
    4. **Drafting**: 파일용 '상세 리포트'와 디스코드용 '요약본'을 작성한다.

    [최종 출력 형식 (Strict Output Format)]
    반드시 아래 태그 형식을 지켜서 출력해.

    <REPORT_FILE>
    # 📅 주간 코드 리뷰 리포트
    ## 1. 총평
    ## 2. 주요 변경 사항 분석
    ## 3. 🚨 발견된 문제점 및 개선 제안
    (여기에 코드 블록과 함께 아주 상세하게 작성해. 마크다운 문법 활용.)
    </REPORT_FILE>

    <DISCORD_EMBED>
    (디스코드 임베드용 요약. 500자 이내.)
    - **말투**: 평소의 친근한 말투 유지.
    - **내용**:
      1. 이번 주 변경된 파일 개수 및 주요 작업 요약 (한 줄)
      2. 칭찬할 점 👍 (없으면 생략)
      3. 고쳐야 할 점 🛠️ (가장 치명적인 것 1~2개만)
      4. "상세한 건 파일 열어서 확인해! 피드백 반영 부탁해~ 😉"
    </DISCORD_EMBED>
    """

    try:
        # 코드 리뷰는 긴 문맥 처리가 중요하므로 Pro 모델 사용 권장
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2 # 코드는 창의성보다 정확성이 생명! 온도를 낮춤.
            )
        )
        return {"status": "success", "report": response.text}

    except Exception as e:
        print(f"Code Review Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/analyze-stock")
async def handle_analyze_stock(request: StockAnalyzeRequest):
    """
    1. 사용자의 쿼리("삼성전자")를 티커("005930")로 변환 (AI 이용)
    2. 주가 데이터 수집 (FinanceDataReader)
    3. 관련 뉴스 수집 (Naver API or Google Search)
    4. 차트 이미지 생성
    5. 종합 리포트 작성 (Gemini)
    """
    print(f"Analyze Stock Request: {request.query}")
    
    # 1. 티커 심볼 찾기 (AI에게 물어봄)
    ticker_prompt = f"""
    사용자가 주식 종목을 찾고 있어. 
    Query: "{request.query}"
    
    가장 적절한 'Yahoo Finance' 또는 'KRX' 기준 티커 심볼(Symbol) 하나만 딱 출력해.
    - 한국 주식: 숫자 6자리 (예: 005930)
    - 미국 주식: 알파벳 티커 (예: AAPL, TSLA)
    - 암호화폐: BTC-USD 등
    - 설명 없이 오직 코드만 반환해.
    """
    try:
        ticker_resp = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=ticker_prompt
        )
        ticker = ticker_resp.text.strip().replace(" ", "")
        print(f"Detected Ticker: {ticker}")
    except Exception as e:
        return {"status": "error", "message": f"티커 찾기 실패: {str(e)}"}

    # 2. 데이터 수집 (Blocking 함수이므로 to_thread 사용)
    try:
        # (A) 주가 데이터 (최근 1년)
        start_date = datetime.now() - pd.DateOffset(years=1)
        start_date_str = start_date.strftime('%Y-%m-%d')
        df = await asyncio.to_thread(fdr.DataReader, ticker, start_date_str)
        if df.empty:
            return {"status": "error", "message": f"데이터를 찾을 수 없어 ({ticker})."}
        
        # 최근 데이터 요약 (AI에게 전달용)
        last_price = df.iloc[-1]['Close']
        start_price = df.iloc[0]['Close']
        change_rate = ((last_price - start_price) / start_price) * 100
        
        stock_summary = f"""
        - 종목코드: {ticker}
        - 현재가: {last_price}
        - 기간 변동률: {change_rate:.2f}%
        - 최근 5일 데이터:\n{df.tail(5).to_string()}
        """

        # (B) 뉴스 데이터 (한국 주식이면 네이버, 아니면 생략 or 구글서치)
        news_text = ""
        if ticker.isdigit(): # 한국 주식(숫자 6자리)
            news_list = await asyncio.to_thread(fetch_naver_news, request.query)
            news_text = "\n".join(news_list)
        else:
            news_text = "(해외 주식은 뉴스 API 연동 필요 - 현재는 차트 위주 분석)"

        # (C) 차트 그리기
        base64_chart = await asyncio.to_thread(draw_stock_chart, df, request.query)

    except Exception as e:
        return {"status": "error", "message": f"데이터 처리 중 오류: {str(e)}"}

    # 3. 최종 리포트 작성 (Gemini Pro)
    report_prompt = f"""
    {AI_PERSONA}
    
    너는 지금부터 유능한 '주식 애널리스트'야.
    아래 데이터를 바탕으로 투자자를 위한 브리핑을 작성해 줘.

    [종목 정보]
    {request.query} ({ticker})

    [주가 데이터 요약]
    {stock_summary}

    [최근 관련 뉴스]
    {news_text}

    [작성 가이드]
    1. **현재 상황**: 주가가 상승세인지 하락세인지 직관적으로 설명해.
    2. **주요 이슈**: 뉴스 내용을 바탕으로 호재/악재를 분석해. (뉴스가 없으면 차트 추세 위주로)
    3. **투자 의견**: 매수/매도 추천은 직접 하지 말고, "어떤 점을 주의해서 봐야 하는지" 관전 포인트를 짚어줘.
    4. 차트 이미지는 내가 이미 그렸으니까, 너는 텍스트 설명에 집중해.
    5. 출력은 Markdown 포맷으로 깔끔하게.
    """

    try:
        report_resp = await client.aio.models.generate_content(
            model='gemini-2.5-pro',
            contents=report_prompt
        )
        report_text = report_resp.text
    except Exception as e:
        report_text = f"분석 리포트 생성 실패: {str(e)}"

    return {
        "status": "success",
        "ticker": ticker,
        "report": report_text,
        "chart_image": base64_chart
    }

@app.get("/")
def read_root():
    return {"status": "Python AI Service is running!"}