from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.frame_processing import process_frame, process_action_recognition
import numpy as np
import cv2
import time
import logging
import asyncio
from app.core.models import get_models, get_executor

router = APIRouter()

# 모델과 executor 가져오기
#detect_model, pose_model, stgcn_processor, action_recognition_system = get_models()
#executor = get_executor()

logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    connection_status = False
    frame_count = 0
    models = None

    try:
        if not connection_status:
            await websocket.accept()
            connection_status = True
            logging.info("WebSocket Connected")

        if models is None:
            models = get_models()
        detect_model, pose_model, stgcn_processor, action_recognition_system = models
        executor = get_executor()

        if executor is None:
            logging.error("Executor is None.")
        else:
            logging.info(f"Executor initialized: {executor}, ID: {id(executor)}")

        while True:
            frame_count += 1
            total_start = time.time()

            # 프레임 수신 (WebSocket)
            receive_start = time.time()
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            receive_time = time.time() - receive_start

            # YOLO 처리 비동기 실행
            yolo_future = asyncio.get_running_loop().run_in_executor(
                None,  # 기본 ThreadPoolExecutor 사용
                process_frame, frame, detect_model, pose_model
            )
            response_data, yolo_time = await yolo_future

            if response_data:
                # YOLO 및 스켈레톤 데이터 WebSocket 전송 (비동기)
                asyncio.create_task(websocket.send_json({
                    "type": "skeleton", 
                    "data": response_data
                }))

                # ST-GCN 비동기 처리
                if response_data["detection_results"].get("poses"):
                    asyncio.create_task(
                        process_action_recognition(
                            response_data["detection_results"]["poses"][0],
                            websocket,
                            stgcn_processor,
                            executor
                        )
                    )

            # FPS 계산 및 로깅
            total_time = time.time() - total_start
            if frame_count % 10 == 0:
                fps = 1 / total_time
                print(f"\n=== 프레임 처리 시간 분석 ===")
                print(f"프레임 번호: {frame_count}")
                print(f"1. 이미지 수신 시간: {receive_time*1000:.2f}ms")
                print(f"2. YOLO 처리 시간: {yolo_time*1000:.2f}ms")
                print(f"총 처리 시간: {total_time*1000:.2f}ms")
                print(f"현재 FPS: {fps:.2f}")
                logging.info(
                    f"FPS: {fps:.2f}, YOLO: {yolo_time*1000:.2f}ms, Receive: {receive_time*1000:.2f}ms"
                )

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")
        logging.info("WebSocket Disconnected")
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        logging.info("WebSocket Disconnected")
    finally:
        # 필요한 정리 작업 수행
        try:
            await websocket.close()
        except:
            pass

