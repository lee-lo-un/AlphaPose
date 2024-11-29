# Backend

## 기술 스택
- FastAPI
- WebSocket
- YOLOv8
- ST-GCN++
- Neo4j
- LangChain/GPT


## 설치 및 실행

* 백엔드 폴더이동
cd backend

1. 의존성 설치:

 - cpu 사용자라면
   ```bash
   pip install .[pytorch-cpu]
   pip install .[mmaction]
   ```
 - gpu 사용자라면
    ```bash
   pip install .[pytorch-gpu]
   pip install .[mmaction]
   ```

- 백엔드에서 위의 2개 명령어를 통해 설치 (cpu와 gpu 사용자에따라 둘중 하나 선택)
- gpu설치자는 CUDA 11.8버전으로 환경 세팅 필요

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