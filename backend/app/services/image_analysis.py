from yoloV11.pose_detection import process_single_person_with_objects
from LangChain.action_recognition import ActionRecognitionSystem
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from app.core.models import get_models
from typing import Optional, Dict, List, Union
import logging

action_recognition_system = ActionRecognitionSystem()

def analyze_image(image_data: str, top5_predictions: Optional[List[Dict[str, Union[str, float]]]] = None) -> Dict:
    try:
        # Base64 이미지 디코딩
        image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)
        models = None

        if models is None:
            models = get_models()
            detect_model, pose_model, stgcn_processor, action_recognition_system = models

        # 스켈레톤 및 객체 데이터 추출
        skeleton_data, object_data = process_single_person_with_objects(image_np, detect_model, pose_model)
        
        # 행동 인식
        action_result = None
        similar_actions = []
        action_explanation = None
        
        if skeleton_data:
            action_result = action_recognition_system.process_skeleton_data(
                skeleton_data["keypoints"], object_data
            )
            if action_result:
                similar_actions = action_recognition_system.get_similar_actions(action_result)
                # 행동에 대한 설명 생성
                action_explanation = action_recognition_system.generate_action_explanation(
                    action_result, 
                    skeleton_data, 
                    object_data
                )

        # 분석 결과 구성
        result = {
            "skeleton_data": skeleton_data,
            "object_data": object_data,
            "action_result": action_result,
            "similar_actions": similar_actions,
            "action_explanation": action_explanation,  # 행동 설명 추가
        }
        
        if top5_predictions:
            result["action_predictions"] = top5_predictions
            # 예측 결과와 실제 분석 결과 비교 분석
            result["prediction_analysis"] = action_recognition_system.compare_predictions_with_result(
                top5_predictions, 
                action_result
            ) if action_result else None
            
        return result
        
    except Exception as e:
        logging.error(f"이미지 분석 중 오류 발생: {str(e)}")
        raise Exception(f"이미지 분석 실패: {str(e)}")
