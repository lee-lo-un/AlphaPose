from typing import TypedDict, Annotated, Sequence, Dict, Any, Tuple
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_teddynote.graphs import visualize_graph
from IPython.display import Image, display 
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import threading

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-3.5-turbo")

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "채팅 이력"]
    skeleton_data: Dict
    extracted_features: Dict
    context: Dict
    knowledge_graph: Dict
    current_action: str

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
        visualize_gaze_direction(skeleton)
        
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

def extract_features(state: Dict) -> Dict:
    """스켈레톤 데이터로부터 중요 특징을 추출"""
    try:
        skeleton = state['skeleton_data']
        print("스켈레톤 데이터 상세 분석 중...")
        
        # 모든 관절 데이터 가져오기
        left_wrist = skeleton.get('left_wrist', {})
        right_wrist = skeleton.get('right_wrist', {})
        left_elbow = skeleton.get('left_elbow', {})
        right_elbow = skeleton.get('right_elbow', {})
        left_shoulder = skeleton.get('left_shoulder', {})
        right_shoulder = skeleton.get('right_shoulder', {})
        
        # 손목 간 거리 계산
        wrist_distance = ((left_wrist.get('x', 0) - right_wrist.get('x', 0)) ** 2 + 
                         (left_wrist.get('y', 0) - right_wrist.get('y', 0)) ** 2) ** 0.5
        
        # 팔꿈치 각도 계산
        try:
            left_elbow_angle = calculate_angle(
                left_shoulder.get('x', 0), left_shoulder.get('y', 0),
                left_elbow.get('x', 0), left_elbow.get('y', 0),
                left_wrist.get('x', 0), left_wrist.get('y', 0)
            )
            right_elbow_angle = calculate_angle(
                right_shoulder.get('x', 0), right_shoulder.get('y', 0),
                right_elbow.get('x', 0), right_elbow.get('y', 0),
                right_wrist.get('x', 0), right_wrist.get('y', 0)
            )
        except Exception as angle_error:
            print(f"각도 계산 중 오류: {angle_error}")
            left_elbow_angle = 0
            right_elbow_angle = 0
        
        # 방향 분석 추가
        direction = calculate_direction(skeleton)
        
        # 관절 상태 분석
        joint_states = analyze_joint_states(skeleton)
        
        # 머리 방향 분석 추가
        head_info = analyze_head_direction(skeleton)
        
        # 전체 자세 분석 추가
        posture_info = analyze_posture(skeleton)
        
        features = {
            "손목_간_거리": f"{wrist_distance:.2f}",
            "왼쪽_팔꿈치_각도": f"{left_elbow_angle:.2f}",
            "오른쪽_팔꿈치_각도": f"{right_elbow_angle:.2f}",
            "자세_특징": "양팔을 앞으로 뻗은 자세" if wrist_distance < 50 else "양팔을 벌린 자세",
            "방향": direction,
            "관절_상태": joint_states,
            "머리_정보": head_info,
            "자세_정보": posture_info,
            "yolo_objects": state.get('yolo_objects', []),  # YOLO 탐지 객체
            "st_gcn_result": state.get('st_gcn_result', '')  # ST-GCN 결과
        }
        
        state['extracted_features'] = features
        print(f"추출된 특징: {features}")
        return state
    except Exception as e:
        print(f"특징 추출 중 오류 발생: {e}")
        print(f"현재 스켈레톤 데이터: {skeleton}")  # 디버깅을 위한 데이터 출력
        raise e

def calculate_angle(x1, y1, x2, y2, x3, y3):
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

def analyze_joint_states(skeleton: Dict) -> Dict:
    """각 관절의 상태 분석"""
    joint_states = {}
    
    # 손 위치 분석
    if 'left_hand' in skeleton and 'right_hand' in skeleton:
        hand_distance = calculate_distance(
            skeleton['left_hand'], 
            skeleton['right_hand']
        )
        joint_states['hands_state'] = "맞잡음" if hand_distance < 30 else "벌어짐"
    
    # 다리 상태 분석
    if 'left_foot' in skeleton and 'right_foot' in skeleton:
        left_foot_pos = skeleton['left_foot'].get('y', 0)
        right_foot_pos = skeleton['right_foot'].get('y', 0)
        joint_states['feet_state'] = "왼발 앞" if left_foot_pos > right_foot_pos else "오른발 앞"
    
    return joint_states

def calculate_distance(point1: Dict, point2: Dict) -> float:
    """두 점 사이의 거리 계산"""
    try:
        x1, y1 = point1.get('x', 0), point1.get('y', 0)
        x2, y2 = point2.get('x', 0), point2.get('y', 0)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    except Exception as e:
        print(f"거리 계산 중 오류: {e}")
        return 0.0

def generate_gpt_interpretation(state: Dict) -> Dict:
    """특징을 바탕으로 GPT 해석 요청"""
    try:
        print("\nGPT를 통한 종합적 상황 분석 중...")
        
        features = state['extracted_features']
        features_str = "\n".join([f"{k}: {v}" for k, v in features.items()])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 인간의 동작과 상황을 종합적으로 분석하는 전문가입니다. 
            스켈레톤 데이터, 객체 인식 결과, ST-GCN 분석 결과를 종합하여 현재 상황을 상세히 설명해주세요."""),
            ("user", """다음 정보를 바탕으로 현재 상황을 분석해주세요:
            1. 스켈레톤 특징: {features}
            2. ST-GCN 행동 분석: {st_gcn}
            3. 주변 객체: {objects}""")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "features": features_str,
            "st_gcn": features.get('st_gcn_result', '없음'),
            "objects": features.get('yolo_objects', '없음')
        })
        
        state['current_action'] = response.content
        print(f"GPT 해석 결과: {response.content}")
        return state
    except Exception as e:
        print(f"GPT 해석 중 오류 발: {e}")
        raise e

def update_knowledge_graph(state: Dict) -> Dict:
    """Neo4j 지식 그래프 업데이트"""
    try:
        print("\n지식 그래프 업데이트 중...")
        features = state['extracted_features']
        action = state['current_action']
        
        # Neo4j 업데이트를 위한 데이터 준비
        joint_data = {
            "angles": {
                "left_elbow": float(features.get("왼쪽_팔꿈치_각도", 0)),
                "right_elbow": float(features.get("오른쪽_팔꿈치_각도", 0)),
                "head_yaw": float(features.get("머리_정보", {}).get("머리_좌우_회전", 0)),
                "head_pitch": float(features.get("머리_정보", {}).get("머리_상하_회전", 0))
            },
            "positions": features.get("관절_상태", {}),
            "direction": features.get("방향", "알 수 없음"),
            "posture": features.get("자세_정보", {}),
            "objects": features.get("yolo_objects", [])
        }
        
        # 각도 범위 설정 (±4도)
        for joint, angle in joint_data["angles"].items():
            joint_data["angles"][joint] = {
                "min": angle - 4,
                "max": angle + 4
            }
        
        state['knowledge_graph_data'] = joint_data
        return state
        
    except Exception as e:
        print(f"지식 그래프 업데이트 중 오류 발생: {e}")
        return state

def create_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("extract", extract_features)
    workflow.add_node("interpret", generate_gpt_interpretation)
    workflow.add_node("update", update_knowledge_graph)
    
    # 순차적 실행을 위한 엣지 연결
    workflow.add_edge("extract", "interpret")
    workflow.add_edge("interpret", "update")
    
    # 시작 노드 설정
    workflow.set_entry_point("extract")

    app = workflow.compile()

    return app


def show_graph_popup_cv2(app):
    try:
        def show_image():
            graph_image = app.get_graph(xray=True).draw_mermaid_png()
            image_array = np.frombuffer(graph_image, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            window_name = 'Workflow Graph'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image)
            
            while True:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            cv2.destroyAllWindows()

        # 일반 스레드 사용
        thread = threading.Thread(target=show_image, daemon=False)
        thread.start()
        
        print("그래프 창이 표시되었습니다. ESC 키를 누르거나 창을 닫으면 종료됩니다.")
        
    except Exception as e:
        print(f"이미지 표시 중 오류 발생: {e}")