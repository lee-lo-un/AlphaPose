from app.core.models import get_models, get_executor
from yoloV11.pose_detection import process_realtime
import asyncio
import time
import logging
import pandas as pd
from fastapi import WebSocketDisconnect

def process_frame(frame, detect_model, pose_model):
    try:
        # YOLO 처리 시작
        yolo_start = time.time()
        yolo_results = process_realtime(frame, detect_model, pose_model)
        yolo_time = time.time() - yolo_start

        # 결과 데이터 생성
        response_data = {
            "timestamp": str(pd.Timestamp.now()),
            "detection_results": yolo_results,
        }

        return response_data, yolo_time

    except ValueError as ve:
        logging.error(f"YOLO 처리 값 오류(ValueError): {ve}")
        return None, 0
    except TypeError as te:
        logging.error(f"YOLO 처리 타입 오류(TypeError): {te}")
        return None, 0
    except RuntimeError as re:
        logging.error(f"YOLO 처리 런타임 오류(RuntimeError): {re}")
        return None, 0
    except Exception as e:
        logging.error(f"YOLO 처리에서 알 수 없는 오류 발생: {e}")
        return None, 0


async def process_action_recognition(pose, websocket, stgcn_processor, executor):
    """ST-GCN 처리 및 결과 WebSocket 전송"""
    try:
        # ST-GCN 처리 비동기 실행
        action_result = await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: stgcn_processor.recognize_action(pose)
        )

        if action_result:
            action_data = {
                "timestamp": str(pd.Timestamp.now()),
                "action_result": action_result,
            }
            try:
                # WebSocket으로 데이터 전송
                await websocket.send_json({"type": "action", "data": action_data})
            except WebSocketDisconnect:
                logging.warning("WebSocket 연결이 종료되었습니다. 데이터 전송 취소.")
                return
            except Exception as send_error:
                logging.error(f"WebSocket 전송 중 오류 발생: {send_error}")
                # 필요 시 재전송 로직 추가 가능
    except WebSocketDisconnect:
        logging.warning("WebSocket 연결이 종료되었습니다. ST-GCN 처리 중단.")
        return
    except Exception as e:
        logging.error(f"ST-GCN 처리 중 오류 발생: {e}")
