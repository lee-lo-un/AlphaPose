from typing import Annotated, TypedDict, Dict, List, Any
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_teddynote.graphs import visualize_graph
from LangChain.workflow_nodes import create_workflow, GraphState, show_graph_popup_cv2
from Neo4j.neo4j_interface import Neo4jInterface
import logging

class State(TypedDict):
    messages: Annotated[list, add_messages]   

class ActionRecognitionSystem:
    def __init__(self):
        self.workflow = create_workflow()
        self.neo4j = Neo4jInterface()
        #show_graph_popup_cv2(self.workflow)
        print("행동 인식 시스템이 초기화되었습니다.")
        
    def process_skeleton_data(self, skeleton_data: Dict[str, Any], 
                            yolo_objects: List = None, 
                            text: str = None,
                            top5_predictions: List = None):
        """스켈레톤 데이터 처리 및 행동 인식"""
        try:
            initial_state = {
                "messages": [text] if text else [],
                "skeleton_data": skeleton_data,
                "yolo_objects": yolo_objects or [],
                "st_gcn_result": "",
                "extracted_features": {},
                "context": {},
                "knowledge_graph": {},
                "current_action": "",
                "top5_predictions": top5_predictions
            }
            
            # 워크플로우 실행
            final_state = self.workflow.invoke(initial_state)
            print("=========final_state3333========>", final_state.get('top5_predictions'))
            # Neo4j 업데이트
            #self.neo4j.update_action_knowledge(
            #    final_state['extracted_features'],
            #    final_state['current_action']
            #)
            print("=========final_state4444========>", final_state.get('top5_predictions'))
            return {
                'action': final_state.get('current_action'),
                'gpt_interpretation': final_state.get('messages', [])[-1] if final_state.get('messages') else None
            }
                
        except Exception as e:
            print(f"\n워크플로우 실행 중 오류 발생: {e}")
            return None

    def get_similar_actions(self, action: str) -> List[str]:
        """지식 그래프에서 유사 행동 검색"""
        # Neo4j 쿼리를 통한 유사 행동 검색 구현
        # 향후 구현 예정
        return []

    def generate_action_explanation(self, action_result, skeleton_data, object_data):
        """행동에 대한 자세한 설명을 생성합니다."""
        try:
            action = action_result["action"]
            confidence = action_result["top5_predictions"]["confidence"]
            
            # 스켈레톤 데이터로부터 주요 자세 특징 추출
            posture_features = self.extract_posture_features(skeleton_data)
            
            # 객체 상호작용 분석
            object_interactions = self.analyze_object_interactions(object_data)
            
            explanation = {
                "main_action": action,
                "confidence_level": confidence,
                "posture_description": posture_features,
                "object_interactions": object_interactions,
                "detailed_explanation": f"{action} 동작이 감지되었습니다. "
                                     f"이 동작은 {posture_features.get('main_characteristics', '')}의 특징을 보이며, "
                                     f"신뢰도는 {confidence*100:.1f}%입니다."
            }
            
            return explanation
        except Exception as e:
            logging.error(f"설명 생성 중 오류 발생: {str(e)}")
            return None

    def compare_predictions_with_result(self, top5_predictions, actual_result):
        """실시간 예측과 최종 분석 결과를 비교 분석합니다."""
        try:
            actual_action = actual_result["action"]
            
            analysis = {
                "match_found": False,
                "prediction_accuracy": 0,
                "ranking": None,
                "comparison": []
            }
            
            for idx, pred in enumerate(top5_predictions):
                if pred["action"] == actual_action:
                    analysis["match_found"] = True
                    analysis["prediction_accuracy"] = pred["confidence"]
                    analysis["ranking"] = idx + 1
                    break
                    
            analysis["comparison"] = [
                {
                    "predicted": pred["action"],
                    "confidence": pred["confidence"],
                    "matches_actual": pred["action"] == actual_action
                }
                for pred in top5_predictions
            ]
            
            return analysis
        except Exception as e:
            logging.error(f"예측 비교 분석 중 오류 발생: {str(e)}")
            return None