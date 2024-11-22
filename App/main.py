from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from LangChain.action_recognition import ActionRecognitionSystem
from YoloV8.pose_detection2 import process_realtime, initialize_models
import traceback

app = FastAPI()

class ImageData(BaseModel):
    image: str

# 모델 초기화
detect_model, pose_model = initialize_models()
action_recognition_system = ActionRecognitionSystem()

@app.post("/analyze")
async def analyze_image(data: ImageData):
    try:
        # 1. Base64 이미지 디코딩
        image_data = data.image.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)

        # 2. 스켈레톤 및 객체 데이터 추출
        print("스켈레톤 및 객체 데이터를 추출합니다...")
        results = process_realtime(image_np, detect_model, pose_model)
        skeleton_data = results.get('poses', [])
        object_data = results.get('objects', [])

        # 3. 행동 인식 (스켈레톤 데이터가 있을 경우)
        if skeleton_data:
            print("행동을 분석합니다...")
            action_result = action_recognition_system.process_skeleton_data(skeleton_data[0])

            # 유사 행동 검색
            print("유사 행동을 검색합니다...")
            similar_actions = (
                action_recognition_system.get_similar_actions(action_result)
                if action_result else []
            )

            # 결과 반환
            return {
                "skeleton_data": skeleton_data,
                "object_data": object_data,
                "action_result": action_result,
                "similar_actions": similar_actions,
            }
        else:
            # 스켈레톤 데이터가 없을 경우
            return {
                "skeleton_data": skeleton_data,
                "object_data": object_data,
                "action_result": None,
                "similar_actions": [],
            }
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
