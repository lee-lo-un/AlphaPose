from typing import TypedDict, Annotated, Sequence, Dict, Any, Tuple
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_teddynote.graphs import visualize_graph
from IPython.display import Image, display 
from LangChain.feature_extraction import extract_features

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
llm = ChatOpenAI(model="gpt-4o")

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "채팅 이력"]
    skeleton_data: Dict
    extracted_features: Dict
    context: Dict
    knowledge_graph: Dict
    current_action: str
    yolo_objects: list[Dict]
    st_gcn_result: list[str]

def generate_gpt_interpretation(state: Dict) -> Dict:
    """특징을 바탕으로 GPT 해석 요청"""
    try:
        print("\nGPT를 통한 종합적 상황 분석 중...")
        
        features = state['extracted_features']
        st_gcn = state['st_gcn_result']
        objects = state['yolo_objects']
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