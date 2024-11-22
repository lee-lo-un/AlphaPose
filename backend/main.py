from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import pandas as pd
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

# 현재 파일의 위치를 기준으로 상대 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from YoloV8.pose_detection2 import process_realtime, initialize_models
from stgcn_processor import STGCNProcessor

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 모델과 스레드 풀 선언
detect_model = None
pose_model = None
stgcn_processor = None
executor = ThreadPoolExecutor(max_workers=2)

@app.on_event("startup")
async def startup_event():
    """서버 시작시 모델 초기화"""
    global detect_model, pose_model, stgcn_processor
    try:
        print("모델 초기화 시작...")
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        else:
            print("CPU 모드로 실행")
            
        detect_model, pose_model = initialize_models()
        stgcn_processor = STGCNProcessor()
        stgcn_processor.initialize()
        print("모든 모델 초기화 완료!")
    except Exception as e:
        print(f"모델 초기화 중 오류 발생: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료시 리소스 정리"""
    executor.shutdown(wait=True)

async def process_frame(frame):
    """프레임 처리를 위한 비동기 함수"""
    loop = asyncio.get_event_loop()
    
    # YOLO 처리 (스켈레톤 데이터)
    yolo_start = time.time()
    results = await loop.run_in_executor(
        executor, 
        lambda: process_realtime(frame, detect_model, pose_model)
    )
    yolo_time = time.time() - yolo_start

    # 스켈레톤 데이터를 즉시 전송하기 위한 초기 결과
    initial_response = {
        'timestamp': str(pd.Timestamp.now()),
        'detection_results': results,
        'action_recognition': None
    }
    
    # ST-GCN++ 처리는 별도의 태스크로 실행
    if results.get('poses'):
        stgcn_start = time.time()
        action_result = await loop.run_in_executor(
            executor,
            lambda: stgcn_processor.recognize_action(results['poses'][0])
        )
        stgcn_time = time.time() - stgcn_start
        
        # 동작 인식 결과 디버깅
        if action_result:
            print("\n=== 동작 인식 결과 ===")
            print(f"인식된 동작: {action_result['action']}")
            print(f"신뢰도: {action_result['confidence']*100:.1f}%")
            print("="*30)
            initial_response['action_recognition'] = action_result
    else:
        stgcn_time = 0
    
    return initial_response, yolo_time, stgcn_time

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결 수락됨")
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            total_start = time.time()
            
            # 프레임 수신
            receive_start = time.time()
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            receive_time = time.time() - receive_start

            try:
                # 프레임 처리 및 결과 전송
                response_data, yolo_time, stgcn_time = await process_frame(frame)
                
                send_start = time.time()
                await websocket.send_json(response_data)
                send_time = time.time() - send_start
                
                # 총 처리 시간 계산
                total_time = time.time() - total_start
                
                # 매 10프레임마다 처리 시간 출력
                if frame_count % 10 == 0:
                    print("\n=== 프레임 처리 시간 분석 ===")
                    print(f"프레임 번호: {frame_count}")
                    print(f"1. 이미지 수신 시간: {receive_time*1000:.2f}ms")
                    print(f"2. YOLO 처리 시간: {yolo_time*1000:.2f}ms")
                    print(f"3. ST-GCN++ 처리 시간: {stgcn_time*1000:.2f}ms")
                    print(f"4. 결과 전송 시간: {send_time*1000:.2f}ms")
                    print(f"총 처리 시간: {total_time*1000:.2f}ms")
                    print(f"현재 FPS: {1/total_time:.2f}")
                    
                    # 동작 인식 결과 출력
                    if response_data.get('action_recognition'):
                        action = response_data['action_recognition']
                        print(f"인식된 동작: {action['action']}")
                        print(f"신뢰도: {action['confidence']*100:.1f}%")
                    else:
                        print("동작 인식 결과 없음")
                        
                    print("="*30)
                
            except Exception as e:
                print(f"프레임 처리 중 오류 발생: {e}")
                continue
                
    except WebSocketDisconnect:
        print("클라이언트 연결 종료")
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("WebSocket 연결 종료")

if __name__ == "__main__":
    import uvicorn
    print(f"서버 시작... 기본 경로: {project_root}")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        workers=1,
        loop="asyncio"
    ) 