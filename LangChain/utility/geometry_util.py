import math
from typing import Dict
import numpy as np

def calculate_distance(point1: Dict, point2: Dict) -> float:
    """두 점 사이의 거리 계산"""
    try:
        p1 = np.array([point1.get('x', 0), point1.get('y', 0)])
        p2 = np.array([point2.get('x', 0), point2.get('y', 0)])
        return np.linalg.norm(p2 - p1)
    except Exception as e:
        print(f"거리 계산 중 오류: {e}")
        return 0.0


def calculate_angle(a: Dict, b: Dict, c: Dict) -> float:
    """
    세 점 (a, b, c)을 사용해 b에서 만들어지는 각도를 계산.
    a, b, c는 각각 x, y 좌표를 포함하는 딕셔너리.
    """
    try:
        # 좌표를 numpy 배열로 변환
        a = np.array([a.get('x', 0), a.get('y', 0)])
        b = np.array([b.get('x', 0), b.get('y', 0)])
        c = np.array([c.get('x', 0), c.get('y', 0)])

        # 벡터 계산
        ab = a - b  # 벡터 AB
        cb = c - b  # 벡터 CB

        # 벡터 크기 계산
        magnitude_ab = np.linalg.norm(ab)
        magnitude_cb = np.linalg.norm(cb)

        # 분모가 0인지 확인
        if magnitude_ab == 0 or magnitude_cb == 0:
            print("경고: 벡터의 크기가 0입니다.")
            return 0.0

        # 벡터 내적 계산
        dot_product = np.dot(ab, cb)

        # 각도 계산 (라디안 -> 도 단위 변환)
        # np.clip을 사용해 값 범위를 [-1, 1]로 제한
        cos_theta = np.clip(dot_product / (magnitude_ab * magnitude_cb), -1.0, 1.0)
        angle = np.arccos(cos_theta) * 180 / np.pi
        return angle
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