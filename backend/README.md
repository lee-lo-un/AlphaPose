# Backend

## 기술 스택
- FastAPI
- WebSocket
- YOLOv8
- ST-GCN++
- Neo4j
- LangChain/GPT

## 설치 방법

### 1. 가상환경 설정


# 백엔드 실행
cd backend
uvicorn main:app --reload

# -----------------------------
# Alpha Pose Backend

## 설치 및 실행
1. 의존성 설치:
   ```bash
   pip install -e .
   ```

2. 서버 실행:
   ```bash
   python server.py
   ```

## API 엔드포인트
- WebSocket: `ws://localhost:8000/ws`
- 이미지 분석: `POST /analyze`
- 텍스트 처리: `POST /process_text`

## 구조
backend/
├── app/
│ ├── main.py
│ ├── routers/
│ ├── services/
│ ├── models/
│ └── core/
└── server.py