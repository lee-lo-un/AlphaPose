from neo4j import GraphDatabase
from typing import Dict, Any
import os
from dotenv import load_dotenv

class Neo4jInterface:
    def __init__(self):
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not password:
            raise ValueError("NEO4J_PASSWORD가 설정되지 않았습니다.")
            
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def update_action_knowledge(self, features: Dict[str, Any], action: str):
        """행동 지식 업데이트"""
        with self.driver.session() as session:
            # 관절 노드 생성/업데이트
            session.execute_write(self._create_or_update_joints, features)
            # 행동 노드 생성/업데이트
            session.execute_write(self._create_or_update_action, action, features)
            
    def _create_or_update_joints(self, tx, features):
        """관절 노드 생성/업데이트 쿼리"""
        query = """
        MERGE (j:Joint {name: $joint_name})
        SET j.angle_min = $angle_min,
            j.angle_max = $angle_max,
            j.last_updated = datetime()
        """
        
        for joint, angles in features.get("angles", {}).items():
            tx.run(query, 
                  joint_name=joint,
                  angle_min=angles["min"],
                  angle_max=angles["max"])
                  
    def _create_or_update_action(self, tx, action: str, features: Dict):
        """행동 노드 생성/업데이트 쿼리"""
        query = """
        MERGE (a:Action {name: $action_name})
        SET a.direction = $direction,
            a.objects = $objects,
            a.last_updated = datetime()
        WITH a
        MATCH (j:Joint)
        WHERE j.name IN $joint_names
        CREATE (j)-[:PART_OF]->(a)
        """
        
        tx.run(query,
               action_name=action,
               direction=features.get("direction"),
               objects=features.get("yolo_objects", []),
               joint_names=list(features.get("angles", {}).keys()))
               
    def find_similar_actions(self, action: str, angle_tolerance: float = 4.0):
        """유사한 행동 검색"""
        with self.driver.session() as session:
            result = session.execute_read(self._find_similar_actions_query,
                                       action, angle_tolerance)
            return result
            
    def _find_similar_actions_query(self, tx, action: str, angle_tolerance: float):
        query = """
        MATCH (a:Action {name: $action_name})<-[:PART_OF]-(j:Joint)
        WITH a, collect(j) as joints
        MATCH (other:Action)<-[:PART_OF]-(oj:Joint)
        WHERE other.name <> $action_name
        AND all(j IN joints WHERE 
            abs(j.angle_min - oj.angle_min) <= $tolerance
            AND abs(j.angle_max - oj.angle_max) <= $tolerance)
        RETURN other.name as similar_action
        """
        
        result = tx.run(query, 
                       action_name=action,
                       tolerance=angle_tolerance)
        return [record["similar_action"] for record in result] 