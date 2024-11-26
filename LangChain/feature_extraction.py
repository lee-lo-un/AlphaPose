from typing import Dict
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

        # 발 위치 판단
        left_ankle_position = "발을 차는 중" if left_ankle.get('y', 0) < left_knee.get('y', 0) else ""
        right_ankle_position = "발을 차는 중" if right_ankle.get('y', 0) < right_knee.get('y', 0) else ""

        # 방향 분석 추가
        direction = calculate_direction(skeleton)
        
        # 관절 상태 분석
        #joint_states = analyze_joint_states(skeleton)
        
        # 머리 방향 분�� 추가
        head_info = analyze_head_direction(skeleton)
        
        # 전체 자세 분석 추가
        posture_info = analyze_posture(skeleton)

        joint_states = analyze_state(skeleton)

        features = {
            "손목_간_거리": f"{wrist_distance:.2f}",
            "왼쪽_팔꿈치_각도": f"{left_elbow_angle:.2f}",
            "오른쪽_팔꿈치_각도": f"{right_elbow_angle:.2f}",
            "왼쪽_어깨_각도": f"{left_shoulder_angle:.2f}",
            "오른쪽_어깨_각도": f"{right_shoulder_angle:.2f}",
            "왼쪽_고관절_각도": f"{left_hip_angle:.2f}",
            "오른쪽_고관절_각도": f"{right_hip_angle:.2f}",
            "왼쪽_무릎_각도": f"{left_knee_angle:.2f}",
            "오른쪽_무릎_각도": f"{right_knee_angle:.2f}",
            "왼손_위치": left_wrist_position,
            "오른손_위치": right_wrist_position,
            "왼발_상태": left_ankle_position,
            "오른발_상태": right_ankle_position,
            "방향": direction,
            "머리_정보": head_info,
            "자세_정보": posture_info,
            "yolo_objects": state.get('yolo_objects', []),  # YOLO 탐지 객체
            "st_gcn_result": state.get('st_gcn_result', ''),  # ST-GCN 결과
            "다리상태": joint_states
        }

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



def analyze_posture(skeleton: Dict) -> Dict:
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
            walking_state = "서있음"
        elif foot_distance < 60:
            walking_state = "걷는중"
        else:
            walking_state = "뛰는중"
            
        return {
            "보행_상태": walking_state,
            "발_간격": foot_distance
        }
    except Exception as e:
        print(f"자세 분석 중 오류: {e}")
        return {}