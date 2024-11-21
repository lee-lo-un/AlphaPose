from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import threading
from screeninfo import get_monitors  # pip install screeninfo


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

def extract_features(state: Dict) -> Dict:
    """스켈레톤 데이터로부터 중요 특징을 추출"""
    try:
        skeleton = state['skeleton_data']
        print("스켈레톤 데이터 분석 중...")
        
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
        
        features = {
            "손목_간_거리": f"{wrist_distance:.2f}",
            "왼쪽_팔꿈치_각도": f"{left_elbow_angle:.2f}",
            "오른쪽_팔꿈치_각도": f"{right_elbow_angle:.2f}",
            "자세_특징": "양팔을 앞으로 뻗은 자세" if wrist_distance < 50 else "양팔을 벌린 자세"
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

def generate_gpt_interpretation(state: Dict) -> Dict:
    """특징을 바탕으로 GPT 해석 요청"""
    try:
        print("\nGPT를 통한 동작 해석 중...")
        
        features_str = "\n".join([f"{k}: {v}" for k, v in state['extracted_features'].items()])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 인간의 동작을 분석하는 전문가입니다. 주어진 스켈레톤 데이터의 특징을 바탕으로 현재 동작을 해석해주세요."),
            ("user", f"다음 특징들을 분석하여 현재 동작을 설명해주세요:\n{features_str}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"features": features_str})
        
        state['current_action'] = response.content
        print(f"GPT 해석 결과: {response.content}")
        return state
    except Exception as e:
        print(f"GPT 해석 중 오류 발생: {e}")
        raise e

def update_knowledge_graph(state: Dict) -> Dict:
    """Neo4j 지식 그래프 업데이트"""
    try:
        print("\n지식 그래프 업데이트 중...")
        action = state['current_action']
        features = state['extracted_features']
        
        # 여기에 Neo4j 업데이트 로직 구현
        print(f"동작 '{action}' 지식 그래프에 저장됨")
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
    
    return workflow.compile()

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