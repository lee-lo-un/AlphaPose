import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import atexit
import os

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from yoloV11.pose_detection import initialize_models
from backend.stgcn_processor import STGCNProcessor
from LangChain.action_recognition import ActionRecognitionSystem

# 전역 변수로 선언
detect_model = None
pose_model = None
stgcn_processor = None
action_recognition_system = None
#executor = ThreadPoolExecutor(max_workers=4)  # executor 초기화
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

def init_models():
    """모델 초기화"""
    global detect_model, pose_model, stgcn_processor, action_recognition_system
    
    try:
        print("모델 초기화 시작...")
        detect_model, pose_model = initialize_models()
        if not detect_model or not pose_model:
            raise ValueError("YOLO 모델 초기화 실패")
            
        stgcn_processor = STGCNProcessor()
        stgcn_processor.initialize()
        
        action_recognition_system = ActionRecognitionSystem()
        print("모델 초기화 완료!")
        
    except Exception as e:
        logging.error(f"모델 초기화 중 오류 발생: {e}")
        raise

def get_models():
    """모델 인스턴스 반환"""
    if None in (detect_model, pose_model, stgcn_processor, action_recognition_system):
        logging.error("모델이 초기화되지 않았습니다.")
        raise ValueError("모델이 초기화되지 않았습니다.")
    return detect_model, pose_model, stgcn_processor, action_recognition_system

def get_executor():
    """executor 인스턴스 반환"""
    return executor

def cleanup():
    if executor:
        executor.shutdown(wait=True)
        print("스레드 풀 종료 완료.")
        
# 프로그램 종료 시 cleanup 호출
atexit.register(cleanup)