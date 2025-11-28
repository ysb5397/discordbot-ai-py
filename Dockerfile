# /my-python-ai-service/Dockerfile

# 1. 파이썬 3.11 버전을 기반으로 시작
FROM python:3.11-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 라이브러리 목록 복사 및 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 모든 소스 코드 복사
COPY . .

# 5. gunicorn 서버 실행
# --timeout-keep-alive 300: 연결 유지 시간을 300초(5분)로 설정
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "--workers", "2", "--max-requests", "1000", "--max-requests-jitter", "50", "--timeout", "120"]