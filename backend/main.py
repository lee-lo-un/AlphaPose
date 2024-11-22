from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
import logging
from io import BytesIO
from PIL import Image
import base64
import traceback

# 현재 파일의 위치를 기준으로 상대 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from YoloV11.pose_detection import process_realtime, initialize_models, process_single_person_with_objects
from stgcn_processor import STGCNProcessor
from LangChain.action_recognition import ActionRecognitionSystem

# FastAPI 앱 설정
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(filename="server_performance.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# 모델 초기화
detect_model, pose_model = initialize_models()
stgcn_processor = STGCNProcessor()
stgcn_processor.initialize()
action_recognition_system = ActionRecognitionSystem()

executor = ThreadPoolExecutor(max_workers=4)  # 스레드 풀 크기 증가

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 초기화"""
    print("모델 초기화 완료!")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 리소스 정리"""
    executor.shutdown(wait=True)
    print("스레드 풀 종료 완료.")

# 이미지 데이터를 처리하기 위한 모델
class ImageData(BaseModel):
    image: str

@app.post("/analyze")
async def analyze_image(data: ImageData):
    """POST 요청으로 이미지 분석"""
    try:
        # Base64 이미지 디코딩
        image_data = data.image.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)
        
        # 이미지 디코딩 후 확인
        #image.save("img/test_input.jpg")  # 디코딩된 이미지를 저장해 확인

        # 스켈레톤 및 객체 데이터 추출
        #results = process_realtime(image_np, detect_model, pose_model)
        #skeleton_data = results.get('poses', [])
        #object_data = results.get('objects', [])
        skeleton_data, object_data = process_single_person_with_objects(image_np, detect_model, pose_model)    
        # 행동 인식
        action_result = None
        similar_actions = []
        print("====skeleton_data: ", skeleton_data)
        print("skeleton_data['keypoints']: ", skeleton_data["keypoints"])
        if skeleton_data:
            action_result = action_recognition_system.process_skeleton_data(skeleton_data["keypoints"])
            if action_result:
                similar_actions = action_recognition_system.get_similar_actions(action_result)
        #response_data, yolo_time = await process_frame(1)
        #print("===response_data: ",response_data)  
        return {
            "skeleton_data": skeleton_data,
            "object_data": object_data,
            "action_result": action_result,
            "similar_actions": similar_actions,
        }
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def process_frame(frame):
    """YOLO와 ST-GCN을 사용한 프레임 처리"""
    loop = asyncio.get_event_loop()
    
    # YOLO 처리
    yolo_start = time.time()
    yolo_results = await loop.run_in_executor(
        executor,
        lambda: process_realtime(frame, detect_model, pose_model)
    )
    yolo_time = time.time() - yolo_start

    # YOLO 결과 반환
    response_data = {
        'timestamp': str(pd.Timestamp.now()),
        'detection_results': yolo_results,
        'action_recognition': None
    }

    # ST-GCN 동작 인식 처리 (비동기 태스크로 실행)
    if yolo_results.get('poses'):
        asyncio.create_task(process_action_recognition(yolo_results['poses'], response_data))

    return response_data, yolo_time

async def process_action_recognition(poses, response_data):
    """ST-GCN 동작 인식을 별도로 처리"""
    loop = asyncio.get_event_loop()
    stgcn_start = time.time()
    action_result = await loop.run_in_executor(
        executor,
        lambda: stgcn_processor.recognize_action(poses[0])
    )
    stgcn_time = time.time() - stgcn_start

    if action_result:
        response_data['action_recognition'] = action_result
        logging.info(f"ST-GCN 처리 시간: {stgcn_time*1000:.2f}ms, 결과: {action_result}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 엔드포인트"""
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

            # 프레임 처리
            response_data, yolo_time = await process_frame(frame)

            # 결과 전송
            send_start = time.time()
            await websocket.send_json(response_data)
            send_time = time.time() - send_start

            # 총 처리 시간 계산
            total_time = time.time() - total_start

            # 매 10프레임마다 처리 시간 출력
            if frame_count % 10 == 0:
                fps = 1 / total_time
                print(f"\n=== 프레임 처리 시간 분석 ===")
                print(f"프레임 번호: {frame_count}")
                print(f"1. 이미지 수신 시간: {receive_time*1000:.2f}ms")
                print(f"2. YOLO 처리 시간: {yolo_time*1000:.2f}ms")
                print(f"3. 결과 전송 시간: {send_time*1000:.2f}ms")
                print(f"총 처리 시간: {total_time*1000:.2f}ms")
                print(f"현재 FPS: {fps:.2f}")
                logging.info(f"FPS: {fps:.2f}, YOLO: {yolo_time*1000:.2f}ms, Receive: {receive_time*1000:.2f}ms, Send: {send_time*1000:.2f}ms")

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

