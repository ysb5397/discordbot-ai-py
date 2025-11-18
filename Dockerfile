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

# 5. Uvicorn 서버 실행 (Express의 app.listen 같은 거)
# 0.0.0.0으로 Koyeb의 모든 요청을 받고, 8000 포트를 사용
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]