import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LangChain.action_recognition import ActionRecognitionSystem
from typing import Dict
import json

def load_sample_skeleton_data() -> Dict:
    """테스트용 샘플 스켈레톤 데이터 로드"""
    sample_data = {
        "keypoints": {
            "left_wrist": {"x": 100, "y": 150, "z": 0},
            "right_wrist": {"x": 120, "y": 155, "z": 0},
            "left_elbow": {"x": 80, "y": 120, "z": 0},
            "right_elbow": {"x": 140, "y": 125, "z": 0},
            "left_shoulder": {"x": 60, "y": 100, "z": 0},
            "right_shoulder": {"x": 160, "y": 100, "z": 0}
        },
        "confidence_scores": {
            "left_wrist": 0.95,
            "right_wrist": 0.93,
            "left_elbow": 0.97,
            "right_elbow": 0.96,
            "left_shoulder": 0.98,
            "right_shoulder": 0.98
        }
    }
    return sample_data["keypoints"]  # keypoints만 반환

def main():
    try:
        # 시스템 초기화
        print("행동 인식 시스템을 초기화합니다...")
        system = ActionRecognitionSystem()
        
        # 스켈레톤 데이터 로드
        print("스켈레톤 데이터를 로드합니다...")
        skeleton_data = load_sample_skeleton_data()
        
        # 행동 분석
        print("행동을 분석합니다...")
        result = system.process_skeleton_data(skeleton_data)
        
        # 결과 출력
        if result:
            print(f"\n인식된 행동: {result}")
            
            # 유사 행동 검색
            print("\n유사한 행동을 검색합니다...")
            similar_actions = system.get_similar_actions(result)
            if similar_actions:
                print("유사한 행동들:")
                for action in similar_actions:
                    print(f"- {action}")
            else:
                print("유사한 행동을 찾을 수 없습니다.")
        else:
            print("\n행동 인식에 실패했습니다.")
            
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
