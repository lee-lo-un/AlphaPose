import math
from typing import Dict
import numpy as np

def calculate_distance(point1: Dict, point2: Dict) -> float:
    """두 점 사이의 거리 계산"""
    try:
        x1, y1 = point1.get('x', 0), point1.get('y', 0)
        x2, y2 = point2.get('x', 0), point2.get('y', 0)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    except Exception as e:
        print(f"거리 계산 중 오류: {e}")
        return 0.0


def calculate_angle(point1: Dict, point2: Dict, point3: Dict) -> float:
    """세 점 사이의 각도 계산"""
    try:
        vector1 = [point2['x'] - point1['x'], point2['y'] - point1['y']]
        vector2 = [point3['x'] - point2['x'], point3['y'] - point2['y']]
        
        # 내적 계산
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        
        # 벡터 크기 계산
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        # 각도 계산
        cos_angle = dot_product / (magnitude1 * magnitude2)
        angle = math.acos(min(1, max(-1, cos_angle)))  # 안정성을 위해 값 제한
        return math.degrees(angle)
    except Exception as e:
        print(f"각도 계산 중 오류: {e}")
        return 0.0

def calculate_angle2(x1, y1, x2, y2, x3, y3):
    """세 점 사이의 각도 계산"""
    import math
    
    # 벡터 계산
    vector1 = [x2 - x1, y2 - y1]
    vector2 = [x3 - x2, y3 - y2]
    
    # 내적 계산
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    
    # 벡터의 크기 계산
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    # 각도 계산 (라디안)
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(min(1, max(-1, cos_angle)))
    
    # 각도를 도(degree)로 변환
    return math.degrees(angle)



def calculate_direction(skeleton_data: Dict) -> str:
    """인체의 방향 계산"""
    try:
        # 주요 키포인트 추출
        nose = np.array([skeleton_data.get('nose', {}).get('x', 0), 
                        skeleton_data.get('nose', {}).get('y', 0)])
        neck = np.array([skeleton_data.get('neck', {}).get('x', 0), 
                        skeleton_data.get('neck', {}).get('y', 0)])
        
        # 방향 벡터 계산
        face_vector = neck - nose
        angle = np.arctan2(face_vector[1], face_vector[0]) * 180 / np.pi
        
        if -45 <= angle <= 45:
            return "오른쪽"
        elif 45 < angle <= 135:
            return "뒤쪽"
        elif -135 <= angle < -45:
            return "앞쪽"
        else:
            return "왼쪽"
    except Exception as e:
        print(f"방향 계산 중 오류: {e}")
        return "알 수 없음"