import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LangChain.action_recognition import ActionRecognitionSystem
from typing import Dict
import json

def load_sample_skeleton_data() -> Dict:
    # 테스트용 샘플 스켈레톤 데이터 로드
    # 테스트용 스켈레톤 데이터 로드
    sample_data = {
        "keypoints": {
            "nose": {"x": 503.59, "y": 124.42, "z": 0},
            "left_eye": {"x": 518.95, "y": 118.19, "z": 0},
            "right_eye": {"x": 497.36, "y": 111.70, "z": 0},
            "left_ear": {"x": 530.24, "y": 131.10, "z": 0},
            "right_ear": {"x": 478.64, "y": 113.53, "z": 0},
            "left_shoulder": {"x": 521.31, "y": 187.03, "z": 0},
            "right_shoulder": {"x": 459.42, "y": 172.57, "z": 0},
            "left_elbow": {"x": 569.56, "y": 255.82, "z": 0},
            "right_elbow": {"x": 473.57, "y": 254.18, "z": 0},
            "left_wrist": {"x": 614.14, "y": 308.04, "z": 0},
            "right_wrist": {"x": 513.67, "y": 318.28, "z": 0},
            "left_hip": {"x": 484.53, "y": 324.26, "z": 0},
            "right_hip": {"x": 444.36, "y": 322.53, "z": 0},
            "left_knee": {"x": 539.80, "y": 420.43, "z": 0},
            "right_knee": {"x": 525.52, "y": 420.28, "z": 0},
            "left_ankle": {"x": 511.02, "y": 547.56, "z": 0},
            "right_ankle": {"x": 492.88, "y": 548.00, "z": 0},
        },
        "confidence_scores": {
            "nose": 0.995,
            "left_eye": 0.956,
            "right_eye": 0.984,
            "left_ear": 0.607,
            "right_ear": 0.893,
            "left_shoulder": 0.996,
            "right_shoulder": 0.998,
            "left_elbow": 0.980,
            "right_elbow": 0.994,
            "left_wrist": 0.983,
            "right_wrist": 0.995,
            "left_hip": 0.998,
            "right_hip": 0.999,
            "left_knee": 0.996,
            "right_knee": 0.997,
            "left_ankle": 0.975,
            "right_ankle": 0.980,
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
