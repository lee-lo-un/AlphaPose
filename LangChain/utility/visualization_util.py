import cv2
import numpy as np
from typing import List, Dict, Tuple
import os

def draw_skeleton(image: np.ndarray, skeleton: Dict, keypoint_pairs: List[Tuple[str, str]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    스켈레톤 데이터를 이미지에 시각화.
    
    Args:
        image (np.ndarray): 원본 이미지.
        skeleton (Dict): 스켈레톤 데이터. 관절 이름과 좌표가 포함된 딕셔너리.
        keypoint_pairs (List[Tuple[str, str]]): 연결할 관절 쌍.
        color (Tuple[int, int, int]): 선 색상 (B, G, R).
    
    Returns:
        np.ndarray: 스켈레톤이 시각화된 이미지.
    """
    try:
        # 각 관절 표시
        for keypoint, coords in skeleton.items():
            x, y = int(coords.get('x', 0)), int(coords.get('y', 0))
            confidence = coords.get('confidence', 0)
            
            if confidence > 0.5:  # 신뢰도 기준
                cv2.circle(image, (x, y), 5, color, -1)

        # 관절 간 연결 표시
        for start, end in keypoint_pairs:
            start_coords = skeleton.get(start, {})
            end_coords = skeleton.get(end, {})
            
            if start_coords and end_coords:
                x1, y1 = int(start_coords.get('x', 0)), int(start_coords.get('y', 0))
                x2, y2 = int(end_coords.get('x', 0)), int(end_coords.get('y', 0))
                
                if start_coords.get('confidence', 0) > 0.5 and end_coords.get('confidence', 0) > 0.5:
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        return image
    except Exception as e:
        print(f"스켈레톤 시각화 중 오류 발생: {e}")
        return image


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    탐지된 객체의 바운딩 박스를 이미지에 시각화.
    
    Args:
        image (np.ndarray): 원본 이미지.
        detections (List[Dict]): 탐지된 객체 목록. 각 객체는 클래스, 신뢰도, 바운딩 박스를 포함.
        color (Tuple[int, int, int]): 바운딩 박스 색상 (B, G, R).
    
    Returns:
        np.ndarray: 바운딩 박스가 시각화된 이미지.
    """
    try:
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            label = detection.get('class', 'Unknown')
            confidence = detection.get('confidence', 0)

            # 바운딩 박스 좌표
            x1, y1, x2, y2 = map(int, bbox)

            # 바운딩 박스와 텍스트 추가
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image
    except Exception as e:
        print(f"바운딩 박스 시각화 중 오류 발생: {e}")
        return image


def overlay_text(image: np.ndarray, text: str, position: Tuple[int, int] = (10, 30), font_scale: float = 1, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    이미지를 기반으로 텍스트 오버레이를 추가.
    
    Args:
        image (np.ndarray): 원본 이미지.
        text (str): 추가할 텍스트.
        position (Tuple[int, int]): 텍스트 위치 (x, y).
        font_scale (float): 텍스트 크기.
        color (Tuple[int, int, int]): 텍스트 색상 (B, G, R).
    
    Returns:
        np.ndarray: 텍스트가 추가된 이미지.
    """
    try:
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return image
    except Exception as e:
        print(f"텍스트 오버레이 중 오류 발생: {e}")
        return image


def save_or_display_image(image: np.ndarray, save_path: str = None):
    """
    시각화된 이미지를 저장하거나 표시.
    
    Args:
        image (np.ndarray): 시각화된 이미지.
        save_path (str, optional): 저장할 경로. 지정하지 않으면 이미지를 화면에 표시.
    """
    try:
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"이미지가 {save_path}에 저장되었습니다.")
        else:
            cv2.imshow("Visualization", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"이미지 저장/표시 중 오류 발생: {e}")


def _classify_head_direction(angle: float) -> str:
    """각도를 바탕으로 머리 방향 분류"""
    if -45 <= angle <= 45:
        return "오른쪽"
    elif 45 < angle <= 135:
        return "뒤쪽"
    elif -135 <= angle < -45:
        return "앞쪽"
    else:
        return "왼쪽"


def visualize_gaze_direction(skeleton: Dict, image_path: str = None):
    """시선 방향을 이미지에 시각화"""
    try:
        # 기본 이미지 경로 설정
        if image_path is None:
            # 현재 파일(workflow_nodes.py)의 위치를 기준으로 상대 경로 계산
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            image_path = os.path.join(project_root, "AlphaPose", "YoloV8", "img", "women_apple.jpg")
        
        print(f"이미지 경로: {image_path}")
        
        # 이미지 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
            
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 키포인트 추출 및 유효성 검사
        nose = np.array([skeleton.get('nose', {}).get('x', 0), 
                        skeleton.get('nose', {}).get('y', 0)])
        left_eye = np.array([skeleton.get('left_eye', {}).get('x', 0), 
                           skeleton.get('left_eye', {}).get('y', 0)])
        right_eye = np.array([skeleton.get('right_eye', {}).get('x', 0), 
                            skeleton.get('right_eye', {}).get('y', 0)])
        neck = np.array([skeleton.get('neck', {}).get('x', 0), 
                        skeleton.get('neck', {}).get('y', 0)])
        
        # 모든 좌표가 0인지 확인
        if np.all(nose == 0) or np.all(left_eye == 0) or np.all(right_eye == 0) or np.all(neck == 0):
            print("유효하지 않은 키포인트가 감지되었습니다.")
            return None
        
        # 눈 중심점 계산
        eye_center = (left_eye + right_eye) / 2
        
        # 시선 벡터 계산 및 정규화
        gaze_vector = eye_center - nose
        gaze_norm = np.linalg.norm(gaze_vector)
        
        # 벡터의 크기가 0인 경우 처리
        if gaze_norm < 1e-10:  # 매우 작은 값으로 체크
            print("시선 벡터의 크기가 너무 작습니다.")
            return None
            
        gaze_vector = gaze_vector / gaze_norm
        
        # 시선 연장선 계산
        extension_length = 100  # 시선 선의 길이
        gaze_endpoint = nose + gaze_vector * extension_length
        
        # 좌표를 정수로 변환하기 전에 범위 체크
        def safe_int_tuple(point):
            return (int(np.clip(point[0], 0, image.shape[1]-1)),
                   int(np.clip(point[1], 0, image.shape[0]-1)))
        
        # 이미지에 시각화
        cv2.circle(image, safe_int_tuple(nose), 5, (0, 0, 255), -1)
        cv2.circle(image, safe_int_tuple(left_eye), 5, (255, 0, 0), -1)
        cv2.circle(image, safe_int_tuple(right_eye), 5, (255, 0, 0), -1)
        cv2.circle(image, safe_int_tuple(neck), 5, (0, 255, 0), -1)
        
        # 시선 방향 선 그리기
        cv2.line(image, 
                safe_int_tuple(nose),
                safe_int_tuple(gaze_endpoint),
                (0, 255, 255), 2)
        
        # 방향 텍스트 추가
        angle = np.arctan2(gaze_vector[1], gaze_vector[0]) * 180 / np.pi
        direction = _classify_head_direction(angle)
        cv2.putText(image, 
                   f"Direction: {direction}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (255, 255, 255), 
                   2)
        
        # 이미지 표시
        cv2.imshow("Gaze Direction Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return gaze_vector
        
    except Exception as e:
        print(f"시선 방향 시각화 중 오류 발생: {e}")
        return None