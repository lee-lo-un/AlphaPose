from typing import TypedDict, Annotated, Sequence, Dict, Any, Tuple, Optional, List, Union
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
    top5_predictions: Optional[List[Dict[str, Union[str, float]]]]

def generate_gpt_interpretation(state: Dict) -> Dict:
    """특징을 바탕으로 GPT 해석 요청"""
    try:
        print("\nGPT를 통한 종합적 상황 분석 중...")
        
        features = state['extracted_features']
        st_gcn = state['st_gcn_result']
        objects = state['yolo_objects']
        messages = state.get('messages', [])
        user_input = messages[0] if messages else ""
        top5_predictions = state.get('top5_predictions', [])
        
        # top5_predictions 정보를 문자열로 변환
        predictions_str = "\n".join([
            f"{pred['action']}: {pred['confidence']*100:.1f}%" 
            for pred in top5_predictions
        ]) if top5_predictions else "예측 정보 없음"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 인간의 동작과 상황을 종합적으로 분석하는 전문가입니다. 
            스켈레톤 데이터, 객체 인식 결과, ST-GCN 분석 결과를 종합하여 현재 상황을 상세히 설명해주세요.
            다만 ST-GCN 분석 결과는 예측치이므로 신뢰도는 낮습니다 참고로만 봐줘야하고 무슨행동일거라는 판단은 관절간 각도 데이터를 기반으로 종합판단해야합니다 
            또한 관절간 각도 데이터는 오차가 있을 수 있으므로 오차범위는 ±4도로 간주해야하고 관절간 각도는 상체와 하체를 먼저 분리해서 판단해야하며 각 팔(어깨-팔꿈치-손목끝간 관절)과 발(무릎-발목-발끝간 관절)로 이어지는 부분으로 나누어 
            현재 인체의 상태를 판단해야합니다 (손의 경우 팔꿈치가 어깨보다 내려가있지만 손목이 위로 가있다면 위로 꺽여있는 모습이므로, 팔의 형태는 들어올리는 형태인것처럼 이어지는 각도들을 종합해서 유추를 잘해야합니다)
            그리고 객체인식 되었다면 3. 4. 과정을 진행하는데 인식결과를 종합하고 사람간의 위치관계도 고려해주면 좋습니다 
            3. 객체_거리_판단에서 나오는 object_class: person는 본인 자신이기 때문에 joint_name은 다른사람이 아닌 나의 스켈레톤 데이터이고 이것은 나이기에 다른사람이 아니라 나의 "joint_name"이라고 판단해야합니다, 그 포인트로부터 탐색된 물체와의 거리와 가깝다면 보고 판단해서 추론할수 있도록합니다, '다른 사람'이 아닌 나자신 본인입니다
            그리고 [object_distance]이것은 나의 손,발과 객체의 거리 판단으로 어떤 상호작용이있는지 보는것이고 [object_interactions]는 나의 중심점과 물체간의 거리판단으로 나와 가까이있는지 판단합니다
            'is_close'는 근접여부 판단 'relative_position'은 나의 중심으로 부터 어느방향에 위치하는지 판단등의 정보로부터 판단할수 있습니다 
            4. 과정이 된다면 근접하게 바라보는 수치도 있으니 어느정도로 보고있을거라는 고려도 할수있습니다 (백팩이나 가방같은 맬수있는것은 몸에 있다면 들고있을수도있지만 손목과의 거리가 엄청 가까워야 들고있는것이고 그렇지않다면 메고있는 상태일수도 있습니다)
            6. 1~5 종합적으로 모든것을 고려해서 자세히 분석해주세요 (다시한번 3번의 joint_name와 물체의 관계는 다른사람이 아닌 나자신과 물체의 관계이기에 다른사람이라고 인식되어서는 안됩니다) 
            7. 최종적으로 2~3줄로 정도로 요약 판단글을 써주세요 (객체들, 사람들간의 위치나 관계 등등 종합적으로 무엇을 하고있고 어떤상황인지 판단. 단, 나의 스켈레톤을 다른사람의 스켈레톤으로 착각하면 안됩니다)    
             """),
            ("user", """다음 정보를 바탕으로 현재 상황을 분석해주세요:
            사용자 입력: {massage}
             
            1. 스켈레톤 상태분석(관절간 각도 데이터를 바탕으로): {features}
            2. 스켈레톤 행동, 데이터 분석: {features}
            3. 주변 객체: {objects}, {object_distance}, {object_interactions}
            4. 시선 객체 교차 확인: {gaze_object}
            5. ST-GCN 분석 결과 예측: {predictions}
            6. 종합판단: 
            7. 최종 요약: """)
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "features": features,
            "st_gcn": features.get('st_gcn_result', '없음'),
            "objects": features.get('yolo_objects', '없음'),
            "object_distance": features.get('객체_거리_판단', '없음'),
            "object_interactions": features.get('중심_객체_거리','없음'),
            "massage": user_input,
            "predictions": predictions_str,
            "gaze_object": features.get('시선_객체_교차', '없음')
        })
        
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