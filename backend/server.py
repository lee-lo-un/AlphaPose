import uvicorn
import signal
import sys
from app.core.config import settings

def signal_handler(sig, frame):
    print('서버를 종료합니다...')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print(f"Starting {settings.PROJECT_NAME}")
    print(f"Project root: {settings.PROJECT_ROOT}")
    
    uvicorn.run(
        "app.main:app",        # 문자열 경로로 변경
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        workers=settings.WORKERS,
        loop=settings.LOOP_TYPE
    )