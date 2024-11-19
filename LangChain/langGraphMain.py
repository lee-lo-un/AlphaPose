from typing import Dict, TypedDict, List, Tuple, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, MessageGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase  # Neo4j 직접 사용
from langchain_core.output_parsers import StrOutputParser
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상태 타입 정의
class GraphState(TypedDict):
    messages: List[BaseMessage]
    context: List[str]
    skeleton_features: Dict[str, Any]
    current_action: str
    neo4j_results: List[Dict]
    final_response: str

# Neo4j 클래스 추가
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

# 특징 추출 노드
def process_skeleton_features(state: GraphState) -> GraphState:
    """스켈레톤 특징을 처리하고 텍스트화"""
    features = state["skeleton_features"]
    processed_features = {
        "joint_angles": features.get("angles", {}),
        "distances": features.get("distances", {}),
        "object_relations": features.get("relations", {})
    }
    state["context"].append(f"스켈레톤 특징: {processed_features}")
    return state
# Neo4j 검색 노드
def search_similar_actions(state: GraphState) -> GraphState:
    """유사 행동 패턴 검색"""
    try:
        # Neo4j 연결 설정
        neo4j_conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            user="neo4j",
            password="your_password"
        )
        
        features = state["skeleton_features"]
        # 실제 벡터 검색 쿼리 구현
        query = """
        MATCH (a:Action)
        WHERE ... // 벡터 유사도 검색 조건
        RETURN a
        LIMIT 5
        """
        
        state["neo4j_results"] = neo4j_conn.query(query)
        neo4j_conn.close()
        
    except Exception as e:
        print(f"Neo4j 검색 중 오류 발생: {e}")
        state["neo4j_results"] = []
        
    return state

# GPT 해석 노드
def interpret_action(state: GraphState) -> GraphState:
    """행동 해석"""
    llm = ChatOpenAI(temperature=0)
    
    context = "\n".join(state["context"])
    neo4j_context = "\n".join([str(r) for r in state["neo4j_results"]])
    
    prompt = f"""
    주어진 스켈레톤 데이터와 과거 유사 행동 기록을 바탕으로 현재 행동을 분석해주세요.
    
    [스켈레톤 데이터]
    {context}
    
    [과거 유사 행동]
    {neo4j_context}
    
    분석 시 다음 사항을 고려해주세요:
    1. 관절 각도와 거리 정보의 의미
    2. 신체 부위간의 상대적 위치 관계
    3. 과거 유사 행동과의 연관성
    4. 가능한 행동의 의도나 목적
    
    상세한 행동 설명을 제공해주세요.
    """
    
    response = llm.invoke(prompt)
    state["current_action"] = response.content
    return state

# 맥락 통합 노드
def integrate_context(state: GraphState) -> GraphState:
    """맥락 정보 통합"""
    llm = ChatOpenAI(temperature=0)
    
    prompt = f"""
    다음 정보들을 통합하여 최종 상황을 해석해주세요:
    
    현재 행동: {state["current_action"]}
    스켈레톤 특징: {state["context"]}
    유사 행동 기록: {state["neo4j_results"]}
    """
    
    response = llm.invoke(prompt)
    state["final_response"] = response.content
    return state

# 그래프 구성
def create_graph() -> Graph:
    # 워크플로우 정의
    workflow = Graph()
    
    # 노드 추가
    workflow.add_node("process_features", process_skeleton_features)
    workflow.add_node("search_actions", search_similar_actions)
    workflow.add_node("interpret", interpret_action)
    workflow.add_node("integrate", integrate_context)
    
    # 엣지 연결
    workflow.add_edge("process_features", "search_actions")
    workflow.add_edge("search_actions", "interpret")
    workflow.add_edge("interpret", "integrate")
    
    return workflow.compile()

# 실행 함수
def run_rag_system(skeleton_features: Dict[str, Any]) -> Optional[str]:
    """RAG 시스템 실행"""
    try:
        graph = create_graph()
        
        initial_state = GraphState(
            messages=[],
            context=[],
            skeleton_features=skeleton_features,
            current_action="",
            neo4j_results=[],
            final_response=""
        )
        
        logger.info("그래프 실행 시작")
        final_state = graph.invoke(initial_state)
        logger.info("그래프 실행 완료")
        
        return final_state["final_response"]
        
    except Exception as e:
        logger.error(f"RAG 시스템 실행 중 오류 발생: {e}")
        return None

# 사용 예시
if __name__ == "__main__":
    # 테스트용 스켈레톤 특징
    test_features = {
        "angles": {
            "elbow_angle": 90,
            "wrist_distance": 10
        },
        "distances": {
            "hands_distance": 15
        },
        "relations": {
            "hand_position": "above_head"
        }
    }
    
    result = run_rag_system(test_features)
    print("최종 해석:", result)
