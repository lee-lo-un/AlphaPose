from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Alpha Pose Backend"
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    
    # FastAPI 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    WORKERS: int = 1
    LOOP_TYPE: str = "asyncio"
    LOG_LEVEL: str = "debug"

    # 로깅 설정
    LOGGING_FILE: str = "server_performance.log"
    
    # 모델 경로 설정
    MODEL_PATH: Path = PROJECT_ROOT / "models"
    
    # YOLO 설정
    YOLO_CONFIDENCE: float = 0.5
    
    # WebSocket 설정
    WS_PING_INTERVAL: int = 30
    
    class Config:
        case_sensitive = True

settings = Settings()