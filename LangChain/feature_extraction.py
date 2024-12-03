from typing import Dict
from click import Tuple
import numpy as np
from LangChain.utility.geometry_util import calculate_angle, calculate_distance, calculate_direction

def extract_features(state: Dict) -> Dict:
    """스켈레톤 데이터로부터 중요 특징을 추출"""
    try:
        skeleton = state['skeleton_data']
        print("스켈레톤 데이터 상세 분석 중...")

        # 주요 관절 데이터 가져오기
        left_wrist = skeleton.get('left_wrist', {})
        right_wrist = skeleton.get('right_wrist', {})
        left_elbow = skeleton.get('left_elbow', {})
        right_elbow = skeleton.get('right_elbow', {})
        left_shoulder = skeleton.get('left_shoulder', {})
        right_shoulder = skeleton.get('right_shoulder', {})
        left_hip = skeleton.get('left_hip', {})
        right_hip = skeleton.get('right_hip', {})
        left_knee = skeleton.get('left_knee', {})
        right_knee = skeleton.get('right_knee', {})
        left_ankle = skeleton.get('left_ankle', {})
        right_ankle = skeleton.get('right_ankle', {})

        # 손목 간 거리 계산
        wrist_distance = calculate_distance(left_wrist, right_wrist)

        # 팔꿈치 각도 계산
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 어깨 각도 계산
        left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

        # 고관절 각도 계산
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # 무릎 각도 계산
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # 손목 위치 판단
        left_wrist_position = "어깨보다 위" if left_wrist.get('y', 0) < left_shoulder.get('y', 0) else "어깨보다 아래"
        right_wrist_position = "어깨보다 위" if right_wrist.get('y', 0) < right_shoulder.get('y', 0) else "어깨보다 아래"

        # 손목 팔꿈치 위치 판단
        left_wrist_elbow_position = "팔꿈치보다 위" if left_wrist.get('y', 0) < left_elbow.get('y', 0) else "팔꿈치보다 아래"
        right_wrist_elbow_position = "팔꿈치보다 위" if right_wrist.get('y', 0) < right_elbow.get('y', 0) else "팔꿈치보다 아래"

        # 발 위치 판단
        left_ankle_position = "발을 차는 중" if left_ankle.get('y', 0) < left_knee.get('y', 0) else ""
        right_ankle_position = "발을 차는 중" if right_ankle.get('y', 0) < right_knee.get('y', 0) else ""

        features = {
            "손목_간_거리": wrist_distance,
            "왼쪽_팔꿈치_각도": left_elbow_angle,
            "오른쪽_팔꿈치_각도": right_elbow_angle,
            "왼쪽_어깨_각도": left_shoulder_angle,
            "오른쪽_어깨_각도": right_shoulder_angle,
            "왼쪽_고관절_각도": left_hip_angle,
            "오른쪽_고관절_각도": right_hip_angle,
            "왼쪽_무릎_각도": left_knee_angle,
            "오른쪽_무릎_각도": right_knee_angle,
            "왼쪽_손목_어깨_위치": left_wrist_position,
            "오른쪽_손목_어깨_위치": right_wrist_position,
            "왼쪽_손목_팔꿈치_위치": left_wrist_elbow_position,
            "오른쪽_손목_팔꿈치_위치": right_wrist_elbow_position,
            "왼발_상태": left_ankle_position,
            "오른발_상태": right_ankle_position,
        }

        # 방향 분석 추가
        direction = calculate_direction(skeleton)
        
        # 관절 상태 분석
        #joint_states = analyze_joint_states(skeleton)
        
        # 머리 방향 분�� 추가
        head_info = analyze_head_direction(skeleton)
        
        # 전체 자세 분석 추가
        posture_info = analyze_posture(skeleton, features)

        joint_states = analyze_state(skeleton)

        object_skeleton_distance = []
        contact_results = check_contact(state.get('yolo_objects', []), skeleton, threshold=50)
        for result in contact_results:
            print(f"객체 {result['object_class']}가 {result['joint_name']}과 접촉 (거리: {result['distance']:.2f}, 신뢰도: {result['confidence']:.2f})")
            object_skeleton_distance.append(result)

        object_interactions = analyze_object_interactions(state.get('yolo_objects', []))


        # 시선 벡터 계산
        nose = np.array([skeleton.get('nose', {}).get('x', 0), skeleton.get('nose', {}).get('y', 0)])
        left_eye = np.array([skeleton.get('left_eye', {}).get('x', 0), skeleton.get('left_eye', {}).get('y', 0)])
        right_eye = np.array([skeleton.get('right_eye', {}).get('x', 0), skeleton.get('right_eye', {}).get('y', 0)])
        
        intersected_object = "No gaze data available"
        if np.all(nose) and np.all(left_eye) and np.all(right_eye):
            gaze_data = calculate_gaze_vector(nose, left_eye, right_eye)
            print("시선 벡터:     ",gaze_data)
            # 시선과 객체 교차 확인
            intersected_object = check_gaze_intersection(gaze_data, state.get('yolo_objects', []))
            print("시선 객체 교차:     ",intersected_object)

        yolo_objects = [obj.get('class', '') for obj in state.get('yolo_objects', [])]

        features_update = {
            "방향": direction,
            "머리_정보": head_info,
            "자세_정보": posture_info,
            "다리상태": joint_states,
            "yolo_objects": yolo_objects,  # YOLO 탐지 객체
            "객체_거리_판단": object_skeleton_distance,
            "시선_객체_교차": intersected_object,
            "중심_객체_거리": object_interactions,
            "st_gcn_result": state.get('st_gcn_result', ''),  # ST-GCN 결과
        }
        features.update(features_update)

        state['extracted_features'] = features
        print(f"추출된 특징: {features}")
        return state

    except Exception as e:
        print(f"특징 추출 중 오류 발생: {e}")
        print(f"현재 스켈레톤 데이터: {skeleton}")  # 디버깅을 위한 데이터 출력
        raise e


    
def analyze_head_direction(skeleton: Dict) -> Dict:
    """머리 방향과 회전 분석"""
    try:
        # 필요한 키포인트 추출
        nose = np.array([skeleton.get('nose', {}).get('x', 0), 
                        skeleton.get('nose', {}).get('y', 0)])
        neck = np.array([skeleton.get('neck', {}).get('x', 0), 
                        skeleton.get('neck', {}).get('y', 0)])
        left_eye = np.array([skeleton.get('left_eye', {}).get('x', 0), 
                           skeleton.get('left_eye', {}).get('y', 0)])
        right_eye = np.array([skeleton.get('right_eye', {}).get('x', 0), 
                            skeleton.get('right_eye', {}).get('y', 0)])
        left_ear = np.array([skeleton.get('left_ear', {}).get('x', 0), 
                           skeleton.get('left_ear', {}).get('y', 0)])
        right_ear = np.array([skeleton.get('right_ear', {}).get('x', 0), 
                            skeleton.get('right_ear', {}).get('y', 0)])
    
        if nose.size == 0 or (left_eye.size == 0 and right_eye.size == 0):
            # 뒷모습으로 간주
            return {"머리_방향": "뒤쪽", "상세정보": "얼굴 데이터 누락"}
        
        # 눈 중심점 계산
        eye_center = (left_eye + right_eye) / 2
        
        # 시선 벡터 계산
        gaze_vector = eye_center - nose
        gaze_angle = np.arctan2(gaze_vector[1], gaze_vector[0]) * 180 / np.pi
        
        # 머리 회전 계산
        neck_to_nose = nose - neck
        yaw_angle = np.arctan2(neck_to_nose[1], neck_to_nose[0]) * 180 / np.pi
        pitch_angle = np.arctan2(gaze_vector[1], gaze_vector[0]) * 180 / np.pi
        
        # 시선 방향 시각화
        #visualize_gaze_direction(skeleton)
        
        
        return {
            "시선_방향": gaze_angle,
            "머리_좌우_회전": yaw_angle,
            "머리_상하_회전": pitch_angle,
            "머리_방향": _classify_head_direction(gaze_angle)
        }
    except Exception as e:
        print(f"머리 방향 분석 중 오류: {e}")
        return {}
    

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


def analyze_state(skeleton: Dict) -> Dict:
    """각 관절의 상태 분석"""
    joint_states = {}
    
    # 손 위치 분석
    if 'left_hand' in skeleton and 'right_hand' in skeleton:
        hand_distance = calculate_distance(
            skeleton['left_hand'], 
            skeleton['right_hand']
        )
        joint_states['hands_state'] = "맞잡음에 가까움" if hand_distance < 30 else ""
    
    # 다리 상태 분석
    if 'left_foot' in skeleton and 'right_foot' in skeleton:
        left_foot_x_pos = skeleton['left_foot'].get(0, 'x')
        right_foot_x_pos = skeleton['right_foot'].get(0, 'x')
        left_foot_y_pos = skeleton['left_foot'].get('y', 0)
        right_foot_y_pos = skeleton['right_foot'].get('y', 0)
        if _classify_head_direction(skeleton['head_direction']) == "오른쪽":
            if (left_foot_y_pos < right_foot_y_pos) & (left_foot_x_pos > right_foot_x_pos) :
                joint_states['feet_state'] = "왼발 앞"
            else : "오른발 앞"
        elif _classify_head_direction(skeleton['head_direction']) == "왼쪽":     
            if (left_foot_y_pos > right_foot_y_pos) & (left_foot_x_pos < right_foot_x_pos) :
                joint_states['feet_state'] = "오른발 앞"
            else : "왼발 앞"
        elif _classify_head_direction(skeleton['head_direction']) == "앞쪽":
            if (left_foot_y_pos < right_foot_y_pos) & (left_foot_x_pos > right_foot_x_pos) :
                joint_states['feet_state'] = "왼발 앞"
            else : "오른발 앞"
        elif _classify_head_direction(skeleton['head_direction']) == "뒤쪽":
            if (left_foot_y_pos < right_foot_y_pos) & (left_foot_x_pos < right_foot_x_pos) :
                joint_states['feet_state'] = "왼발 앞" 
            else : "오른발 앞"

    
    return joint_states

def analyze_posture(skeleton: Dict, features: Dict) -> Dict:
    """전체 자세 분석"""
    try:
        # 걸음걸이 분석
        left_foot = np.array([skeleton.get('left_foot', {}).get('x', 0), 
                            skeleton.get('left_foot', {}).get('y', 0)])
        right_foot = np.array([skeleton.get('right_foot', {}).get('x', 0), 
                             skeleton.get('right_foot', {}).get('y', 0)])
        hip_center = np.array([skeleton.get('hip_center', {}).get('x', 0), 
                             skeleton.get('hip_center', {}).get('y', 0)])
        
        # 발 간격
        foot_distance = np.linalg.norm(left_foot - right_foot)
        
        # 보행 상태 판단
        if foot_distance < 30:
            walking_state = "다리사이 간격 좁음"
        elif foot_distance < 60:
            walking_state = "다리사이가 조금 벌어짐"
        else:
            walking_state = ""

        left_knee_angle = features.get('왼쪽_무릎_각도', 0)
        right_knee_angle = features.get('오른쪽_무릎_각도', 0)
        # 자세 판단
        if left_knee_angle < 105 and right_knee_angle < 105:
            posture = "앉아 있을 수 있음"
        elif foot_distance > 100:
            posture = "다리사이거리가 있음"
        else:
            posture = ""

        return {
            "보행_상태": walking_state,
            "발_간격": foot_distance,
            "하체_자세": posture
        }
    except Exception as e:
        print(f"자세 분석 중 오류: {e}")
        return {}


### 객체와 사람의 위치분석

def calculate_object_centroids(yolo_objects):
    """YOLO 탐지 객체의 중심점 계산"""
    centroids = []
    for obj in yolo_objects:
        bbox = obj['bbox']  # [x_min, y_min, x_max, y_max]
        centroid_x = (bbox[0] + bbox[2]) / 2
        centroid_y = (bbox[1] + bbox[3]) / 2
        centroids.append({
            'class': obj['class'],
            'confidence': obj['confidence'],
            'centroid': {'x': centroid_x, 'y': centroid_y}
        })
    return centroids

def calculate_distance_between_points(point1, point2):
    """두 점 사이의 거리 계산"""
    return ((point1['x'] - point2['x']) ** 2 + (point1['y'] - point2['y']) ** 2) ** 0.5

def check_contact(yolo_objects, skeleton, threshold=50):
    """
    YOLO 객체와 스켈레톤 관절 간 접촉 여부를 판단
    threshold: 접촉으로 간주할 거리 임계값
    """
    results = []
    centroids = calculate_object_centroids(yolo_objects)
    
    # 관절 데이터 가져오기
    joints = {
        '왼손목': skeleton.get('left_wrist', {}),
        '오른손목': skeleton.get('right_wrist', {}),
        '왼발목': skeleton.get('left_ankle', {}),
        '오른발목': skeleton.get('right_ankle', {}),
        '코': skeleton.get('nose', {})
    }
    
    # 각 객체 중심점과 관절 간 거리 계산
    for obj in centroids:
        for joint_name, joint_data in joints.items():
            if joint_data:  # 관절 데이터가 존재하는 경우에만 계산
                distance = calculate_distance_between_points(obj['centroid'], joint_data)
                if distance <= threshold:
                    results.append({
                        'object_class': obj['class'],
                        'joint_name': joint_name,
                        'distance': distance,
                        'confidence': obj['confidence'],
                        'contact': True
                    })
    
    return results


### 시선으으로 객체를 보는지 판단

def calculate_gaze_vector(nose, left_eye, right_eye) -> Dict:
    """시선 벡터 계산"""
    try:     
        # 눈 중심점 계산
        eye_center = (left_eye + right_eye) / 2
        
        # 시선 벡터 계산 (눈 중심 → 코 방향)
        gaze_vector = nose - eye_center
        
        # 정규화된 벡터 반환
        norm = np.linalg.norm(gaze_vector)
        if norm == 0:
            raise ValueError("시선 벡터의 크기가 0입니다.")
        gaze_vector_normalized = gaze_vector / norm
        
        return {
            "eye_center": eye_center,
            "nose": nose,
            "gaze_vector": gaze_vector_normalized
        }
    except Exception as e:
        print(f"시선 벡터 계산 중 오류: {e}")
        return None
    
# 객체와 시선의 교차 확인
def check_gaze_intersection(gaze_data: Dict, yolo_objects: list, threshold: float = 50.0) -> str:
    """시선 벡터와 객체의 교차 여부 확인"""
    try:
        eye_center = np.array(gaze_data["eye_center"])
        gaze_vector = np.array(gaze_data["gaze_vector"])
        
        # 이미지 경계를 기준으로 시선 연장
        #max_dim = max(image_dims)
        max_dim = (640, 480)
        gaze_endpoint = eye_center + gaze_vector * max_dim  # 최대 길이로 연장
        
        closest_object = None
        closest_distance = float('inf')  # 초기값을 무한대로 설정
        distances = []  # 모든 객체와의 거리 정보를 저장할 리스트

        for obj in yolo_objects:
            bbox = obj['bbox']
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            # 객체 중심 계산
            obj_center = np.array([(bbox_x1 + bbox_x2) / 2, (bbox_y1 + bbox_y2) / 2])
            
            # 객체 중심과 시선 선분 간 거리 계산
            distance = np.linalg.norm(
                np.cross(gaze_endpoint - eye_center, eye_center - obj_center)
            ) / np.linalg.norm(gaze_endpoint - eye_center)

            distances.append({"object_class": obj['class'], "distance": distance})
            
            # 가장 가까운 객체 업데이트
            if distance < threshold and distance < closest_distance:
                closest_distance = distance
                closest_object = obj

        # 결과 생성
        result = {
            "closest_object": closest_object['class'] if closest_object else "No object in gaze direction",
            "closest_distance": closest_distance if closest_object else None,
            "all_distances": distances  # 모든 객체와의 거리 정보 포함
        }

        return result
    except Exception as e:
        print(f"시선 교차 확인 중 오류: {e}")
        return None



def analyze_object_interactions(yolo_objects, interaction_threshold=0.5, ratio_threshold=0.5):
    """
    YOLO 탐지 객체와 사람 간의 상호작용 판단 및 상대적 위치 분석
    Args:
        yolo_objects: YOLO 탐지된 객체 리스트. 각 객체는 'bbox', 'class', 'confidence'를 포함.
        interaction_threshold: 정규화된 거리 임계값 (사람 키 대비 거리 비율).
        ratio_threshold: 비율 차이 임계값 (사람과 객체의 높이/너비 비율 차이).
    Returns:
        interactions: 상호작용 리스트 또는 예외 메시지
    """
    if not yolo_objects:
        return {"error": "YOLO 객체가 탐지되지 않았습니다."}

    # 중심점 계산 및 추가
    for obj in yolo_objects:
        bbox = obj['bbox']
        centroid_x = (bbox[0] + bbox[2]) / 2
        centroid_y = (bbox[1] + bbox[3]) / 2
        obj['centroid'] = {'x': centroid_x, 'y': centroid_y}

    # 사람 객체 필터링
    human_objects = [obj for obj in yolo_objects if obj['class'] == 'person']
    if not human_objects:
        return {"error": "사람 객체가 탐지되지 않았습니다."}

    interactions = []

    # 사람 객체를 기준으로 다른 객체와의 상호작용 분석
    for human in human_objects:
        human_centroid = human['centroid']
        human_bbox = human['bbox']

        # 사람 크기 계산
        human_height = human_bbox[3] - human_bbox[1]
        human_width = human_bbox[2] - human_bbox[0]
        human_ratio = human_height / human_width if human_width != 0 else float("inf")

        for obj in yolo_objects:
            if obj['class'] == 'person':
                continue  # 자신과의 상호작용은 무시

            # 객체 중심점 및 크기 비율 계산
            obj_centroid = obj['centroid']
            obj_bbox = obj['bbox']

            obj_height = obj_bbox[3] - obj_bbox[1]
            obj_width = obj_bbox[2] - obj_bbox[0]
            obj_ratio = obj_height / obj_width if obj_width != 0 else float("inf")

            # 중심점 거리 계산 및 정규화
            distance = calculate_distance_between_points(human_centroid, obj_centroid)
            normalized_distance = distance / human_height  # 사람 키로 거리 정규화

            # 비율 차이 계산
            ratio_difference = abs(human_ratio - obj_ratio)

            # 근접 여부 판단
            is_close = normalized_distance < interaction_threshold
            is_similar_ratio = ratio_difference < ratio_threshold

            # 상대적 위치 계산
            dx = obj_centroid['x'] - human_centroid['x']
            dy = obj_centroid['y'] - human_centroid['y']

            if abs(dx) > abs(dy):
                relative_position = "오른쪽" if dx > 0 else "왼쪽"
            elif abs(dy) > abs(dx):
                relative_position = "아래쪽" if dy > 0 else "위쪽"
            else:
                relative_position = "대각선"

            # 상호작용 추가
            interactions.append({
                "human_centroid": human_centroid,
                "object_centroid": obj_centroid,
                "object_class": obj['class'],
                "distance": distance,
                "normalized_distance": normalized_distance,
                "is_close": is_close,
                "human_ratio": human_ratio,
                "object_ratio": obj_ratio,
                "ratio_difference": ratio_difference,
                "is_similar_ratio": is_similar_ratio,
                "relative_position": relative_position,
                "interaction": f"사람이 {obj['class']}의 {relative_position}에 있고 {'가까움' if is_close else '멀음'}"
            })

    if not interactions:
        return {"error": "사람 객체와 다른 객체 간의 상호작용이 감지되지 않았습니다."}

    return interactions
