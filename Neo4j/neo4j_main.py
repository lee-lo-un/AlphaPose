from knowledge_graph import KnowledgeGraph
from behavior_graph import BehaviorGraph
from utils import calculate_gaussian_weights
from dotenv import load_dotenv
import os
from datetime import datetime
import json
from pprint import pprint

# .env 파일 로드
load_dotenv()

# Neo4j 연결 정보
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

def initialize_knowledge_graph():
    """Knowledge Graph 초기화 및 기본 구조 생성"""
    kg = KnowledgeGraph(uri, user, password)
    try:
        kg.create_body_structure()
    finally:
        kg.close()

def process_pose_data(skeleton_data, direction=None, action=None, description="", image_url="", related_action=None):
    """
    포즈 데이터를 처리하는 메인 함수
    
    Parameters:
    - skeleton_data: dict, 필수, 각 관절의 각도 데이터
    - direction: str, 선택, 인체의 방향 ("정후면", "측면" 등)
    - action: str, 선택, 행동 이름
    - description: str, 선택, 행동에 대한 설명
    - image_url: str, 선택, 관련 이미지 URL
    - related_action: str, 선택, 연관된 행동
    """
    # 스켈레톤 데이터를 내부 형식으로 변환
    processed_data = {}
    for joint, angle in skeleton_data.items():
        processed_data[joint] = {
            "각도": angle,
            "가중치": 1.0,  # 기본 가중치
        }

    bg = BehaviorGraph(uri, user, password)
    result = None
    
    try:
        if action:
            # 행동이 제공된 경우 노드 생성/업데이트
            bg.create_or_update_behavior(
                behavior_name=action,
                skeleton_data=processed_data,
                direction=direction,
                description=description,
                image_url=image_url
            )
            
            # 저장된 데이터 조회 및 출력
            stored_data = bg.get_behavior_details(action, direction)
            print("\n=== 저장된 행동 데이터 ===")
            pprint(stored_data, width=120, sort_dicts=False)
            result = stored_data
            
        else:
            # 행동이 제공되지 않은 경우 검색만 수행
            search_results = bg.search_behavior_by_angles(processed_data)
            print("\n=== 검색 결과 ===")
            pprint(search_results, width=120)
            result = search_results
            
    finally:
        bg.close()
        
    return result

def main():
    load_dotenv()
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    print("=== Neo4j 연결 ===")
    kg = KnowledgeGraph(uri, user, password)
    bg = BehaviorGraph(uri, user, password)

    print("=== Knowledge Graph 초기화 ===")
    if kg.create_body_structure():
        print("기본 신체 구조가 생성되었습니다.")
    else:
        print("기본 신체 구조가 이미 존재합니다.")

    # 테스트 데이터
    test_data = {
        "행동": "걷기",
        "인체 방향": "정후면",
        "스켈레톤": {
            "LeftShoulder": 45,
            "RightShoulder": 50,
            "LeftElbow": 60,
            "RightElbow": 55,
            "LeftHip": 35,
            "RightHip": 40,
            "LeftKnee": 50,
            "RightKnee": 45,
        },
        "연관 행동": "뛰기"
    }

    # 테스트 실행
    print("\n=== 테스트 데이터 저장 및 조회 ===")
    result1 = process_pose_data(
        skeleton_data=test_data["스켈레톤"],
        direction=test_data["인체 방향"],
        action=test_data["행동"],
        related_action=test_data["연관 행동"]
    )

    # 검색 테스트
    print("\n=== 검색 테스트 ===")
    search_data = {
        "LeftHip": 35,
        "RightHip": 40,
        "LeftKnee": 50,
        "RightKnee": 45
    }
    result2 = process_pose_data(
        skeleton_data=search_data,
        direction=test_data["인체 방향"]
    )
    
    return result1, result2
        
if __name__ == "__main__":
    main()
