from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.models import init_models, cleanup
#from app.routers.websocket_router import router as websocket_router
#from app.routers.image_router import router as image_router
#from app.routers.text_router import router as text_router
#from app.routers import websocket_router, image_router, text_router

app = FastAPI(title=settings.PROJECT_NAME)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
setup_logging()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 초기화"""
    print("서버 시작...")
    init_models()

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 리소스 정리"""
    cleanup()
    print("리소스 정리 완료.")

from app.routers import websocket_router, image_router, text_router

# 라우터 등록
app.include_router(websocket_router.router)
app.include_router(image_router.router, prefix="/analyze", tags=["Image Analysis"])
app.include_router(text_router.router, prefix="/process_text", tags=["Text Processing"])

