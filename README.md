# Human Action Analysis Project

실시간 인체 동작 인식 및 분석 시스템

## 1.프로젝트 구조

## 2. 기술 스택
- FastAPI
- WebSocket
- YOLOv8
- ST-GCN++
- Neo4j
- LangChain/GPT

## 3. 설치 방법
순서대로 설치 방법2

- Conda 환경 생성 (Python 10 버전)
```bash
conda create -n alphapose python=10
conda activate alphapos
```

### setup.py 파일로 설치
### (1) 백엔드 설치
- 1) 백엔드 폴더이동
   ```bash
    cd backend
   ```
   cd backend

- 2) 의존성 설치:

- 백엔드에서 cpu와 gpu 사용자에따라 둘중 하나 선택해서 설치
- gpu설치자는 CUDA 11.8버전으로 환경 세팅 필요

 - cpu 사용자라면 
```bash
   pip install .[pytorch-cpu, mmaction]
```
 - gpu 사용자라면
```bash
   pip install .[pytorch-gpu, mmaction]
```

### (2) 프론트 설치
- 1) 프론트엔드 이동
```bash
    cd frontend
```
- 2) 프론트엔드 설치
```bash
    npm install
```

## 4. 서버 실행
- 백엔드 실행

1) 백엔드 이동 
```bash
   cd backend
```
2) 서버 실행:
```bash
   python server.py
```

- 프론트 실행
1) 백엔드 이동 
```bash
   cd frontend
```
2) 서버 실행:
```bash
   npm run dev
```

#

### 방법2. 가상환경 세팅을 이용한 설치

 - 1) Conda로 설치
 ```bash
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8

conda install cython matplotlib numpy pandas pyyaml ninja tqdm opencv requests pillow
```
 - 2) Pip으로 설치
 ```bash
pip install python-dotenv ultralytics neo4j neo4j-driver langchain langchain-community langchain-openai langgraph openai openmim

pip install mmcv==2.1.0

mim install mmdet mmpose mmaction2

pip install fastapi uvicorn websockets python-multipart pydantic pydantic-settings

```
- 3) 프론트엔드 이동
```bash
cd frontend
```
- 4) 프론트엔드 설치
```bash
npm install
```

