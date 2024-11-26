import logging
import sys
from pathlib import Path
from app.core.config import settings

def setup_logging():
    # 로그 파일 경로 설정
    log_file = Path(settings.PROJECT_ROOT) / "logs" / settings.LOGGING_FILE
    
    # logs 디렉토리가 없으면 생성
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 로깅 포맷 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("Logging system initialized")
    logging.info(f"Log file location: {log_file}")