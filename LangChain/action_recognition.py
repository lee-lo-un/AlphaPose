from typing import Annotated, TypedDict, Dict, List, Any
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_teddynote.graphs import visualize_graph
from LangChain.workflow_nodes import create_workflow, GraphState, show_graph_popup_cv2
from Neo4j.neo4j_interface import Neo4jInterface

class State(TypedDict):
    messages: Annotated[list, add_messages]   

class ActionRecognitionSystem:
    def __init__(self):
        self.workflow = create_workflow()
        self.neo4j = Neo4jInterface()
        show_graph_popup_cv2(self.workflow)
        print("행동 인식 시스템이 초기화되었습니다.")
        
    def process_skeleton_data(self, skeleton_data: Dict[str, Any], 
                            yolo_objects: List = None, 
                            st_gcn_result: str = None):
        """스켈레톤 데이터 처리 및 행동 인식"""
        try:
            initial_state = {
                "messages": [],
                "skeleton_data": skeleton_data,
                "yolo_objects": yolo_objects or [],
                "st_gcn_result": st_gcn_result or "",
                "extracted_features": {},
                "context": {},
                "knowledge_graph": {},
                "current_action": ""
            }
            
            # 워크플로우 실행
            final_state = self.workflow.invoke(initial_state)
            
            # Neo4j 업데이트
            self.neo4j.update_action_knowledge(
                final_state['extracted_features'],
                final_state['current_action']
            )
            
            return final_state.get('current_action')
                
        except Exception as e:
            print(f"\n워크플로우 실행 중 오류 발생: {e}")
            return None

    def get_similar_actions(self, action: str) -> List[str]:
        """지식 그래프에서 유사 행동 검색"""
        # Neo4j 쿼리를 통한 유사 행동 검색 구현
        # 향후 구현 예정
        return []