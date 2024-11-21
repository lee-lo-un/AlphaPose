from typing import Annotated, TypedDict, Dict, List, Any
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_teddynote.graphs import visualize_graph
from .langGraphMain import create_workflow, GraphState

class State(TypedDict):
    messages: Annotated[list, add_messages]   

class ActionRecognitionSystem:
    def __init__(self):
        self.workflow = create_workflow()
        print("행동 인식 시스템이 초기화되었습니다.")
        
    def process_skeleton_data(self, skeleton_data: Dict[str, Any]):
        """스켈레톤 데이터 처리 및 행동 인식"""
        try:
            print("\n스켈레톤 데이터 처리 시작...")
            initial_state = {
                "messages": [],
                "skeleton_data": skeleton_data,
                "extracted_features": {},
                "context": {},
                "knowledge_graph": {},
                "current_action": ""
            }
            
            final_state = self.workflow.invoke(initial_state)
            
            if final_state.get('current_action'):
                result = final_state['current_action']
                print(f"\n최종 인식된 행동: {result}")
                return result
            else:
                print("\n행동 인식 실패")
                return None
                
        except Exception as e:
            print(f"\n워크플로우 실행 중 오류 발생: {e}")
            return None

    def get_similar_actions(self, action: str) -> List[str]:
        """지식 그래프에서 유사 행동 검색"""
        # Neo4j 쿼리를 통한 유사 행동 검색 구현
        # 향후 구현 예정
        return []