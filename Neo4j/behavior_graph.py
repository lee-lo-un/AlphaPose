import numpy as np
from neo4j import GraphDatabase
from datetime import datetime
from utils import update_gaussian_parameters, calculate_angle_range, calculate_gaussian_weights
from config import GAUSSIAN_CONFIG

class BehaviorGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_or_update_behavior(self, behavior_name, skeleton_data, direction=None, description="", image_url=""):
        """
        행동 노드를 생성하거나 업데이트합니다.
        행동과 방향이 모두 있을 때만 노드를 생성/업데이트하고,
        그 외의 경우는 검색만 수행합니다.
        """
        # 행동이나 방향이 없으면 검색만 수행
        if not behavior_name or not direction:
            return self.search_behavior_by_angles(skeleton_data)

        with self.driver.session() as session:
            timestamp = datetime.utcnow().isoformat()
            
            # 같은 행동과 방향을 가진 노드 검색
            existing_nodes = session.execute_write(
                self._find_matching_nodes,
                behavior_name,
                direction
            )

            if existing_nodes:  # 매칭되는 노드가 있으면 데이터 병합
                session.execute_write(
                    self._merge_behavior_data,
                    existing_nodes[0],
                    skeleton_data,
                    description,
                    image_url,
                    timestamp
                )
            else:  # 매칭되는 노드가 없으면 새로 생성
                session.execute_write(
                    self._create_new_behavior_node,
                    behavior_name,
                    skeleton_data,
                    direction,
                    description,
                    image_url,
                    timestamp
                )

    @staticmethod
    def _find_matching_nodes(tx, behavior_name, direction):
        """
        같은 행동과 방향을 가진 노드를 찾습니다.
        """
        query = """
        MATCH (b:Behavior)
        WHERE b.name = $behavior_name AND b.direction = $direction
        RETURN b
        """
        return list(tx.run(query, behavior_name=behavior_name, direction=direction))

    @staticmethod
    def _create_new_behavior_node(tx, behavior_name, skeleton_data, direction, description, image_url, timestamp):
        """
        새로운 행동 노드를 생성합니다.
        """
        query = """
        CREATE (b:Behavior)
        SET b.name = $name,
            b.direction = $direction,
            b.descriptions = [$description],
            b.image_urls = [$image_url],
            b.timestamps = [$timestamp]
        """
        tx.run(query, 
               name=behavior_name,
               direction=direction,
               description=description,
               image_url=image_url,
               timestamp=timestamp)

        # 각 관절별 데이터 추가
        for joint_name, joint_data in skeleton_data.items():
            angle = joint_data["각도"]
            weights = joint_data.get("가중치", [])
            
            joint_query = """
            MATCH (b:Behavior)
            WHERE b.name = $name AND b.direction = $direction
            SET b[$joint_name + '_angles'] = $angles,
                b[$joint_name + '_weights'] = $weights,
                b[$joint_name + '_range'] = $range,
                b[$joint_name + '_mu'] = $mu,
                b[$joint_name + '_sigma'] = $sigma
            """
            
            # 단일 각도를 리스트로 변환
            angles = [angle] if not isinstance(angle, list) else angle
            weights = [weights] if not isinstance(weights, list) else weights
            
            tx.run(joint_query,
                   name=behavior_name,
                   direction=direction,
                   joint_name=joint_name,
                   angles=angles,
                   weights=weights,
                   range=joint_data.get("범위", "[0, 0]"),
                   mu=joint_data.get("μ", angles[0]),
                   sigma=joint_data.get("σ", 2.0))

    @staticmethod
    def _merge_behavior_data(tx, node, skeleton_data, description, image_url, timestamp):
        """
        기존 노드에 새로운 데이터를 병합합니다.
        """
        node_id = node["b"].element_id
        
        # 메타데이터 업데이트
        meta_query = """
        MATCH (b:Behavior)
        WHERE elementId(b) = $node_id
        SET b.descriptions = CASE 
            WHEN NOT $description IN b.descriptions 
            THEN b.descriptions + $description 
            ELSE b.descriptions 
            END,
            b.image_urls = CASE 
            WHEN NOT $image_url IN b.image_urls 
            THEN b.image_urls + $image_url 
            ELSE b.image_urls 
            END,
            b.timestamps = CASE 
            WHEN NOT $timestamp IN b.timestamps 
            THEN b.timestamps + $timestamp 
            ELSE b.timestamps 
            END
        """
        tx.run(meta_query,
               node_id=node_id,
               description=description,
               image_url=image_url,
               timestamp=timestamp)

        # 각 관절별 데이터 병합
        for joint_name, joint_data in skeleton_data.items():
            # 기존 데이터 조회
            query = """
            MATCH (b:Behavior)
            WHERE elementId(b) = $node_id
            RETURN b[$joint_name + '_angles'] as angles,
                   b[$joint_name + '_mu'] as mu,
                   b[$joint_name + '_sigma'] as sigma
            """
            result = tx.run(query, node_id=node_id, joint_name=joint_name).single()
            
            existing_angles = result["angles"] if result and result["angles"] else []
            current_mu = result["mu"] if result and result["mu"] is not None else None
            current_sigma = result["sigma"] if result and result["sigma"] is not None else None
            
            # 새로운 각도 데이터
            new_angle = joint_data["각도"]
            
            # 가우시안 파라미터 업데이트
            new_mu, new_sigma, score = update_gaussian_parameters(
                existing_angles=existing_angles,
                new_angle=new_angle,
                current_mu=current_mu if current_mu is not None else new_angle,
                current_sigma=current_sigma if current_sigma is not None else 5.0
            )
            
            # 새로운 범위 계산
            new_range = calculate_angle_range(new_mu, new_sigma)
            
            # 데이터 업데이트
            merge_query = """
            MATCH (b:Behavior)
            WHERE elementId(b) = $node_id
            SET b[$joint_name + '_angles'] = 
                CASE 
                    WHEN b[$joint_name + '_angles'] IS NULL 
                    THEN $new_angles
                    ELSE b[$joint_name + '_angles'] + $new_angles
                END,
            b[$joint_name + '_weights'] = $score,
            b[$joint_name + '_range'] = $range,
            b[$joint_name + '_mu'] = $mu,
            b[$joint_name + '_sigma'] = $sigma
            """
            
            tx.run(merge_query,
                   node_id=node_id,
                   joint_name=joint_name,
                   new_angles=[new_angle],
                   score=score,
                   range=new_range,
                   mu=new_mu,
                   sigma=new_sigma)

    def search_behavior_by_angles(self, skeleton_data):
        """
        관절 각도로부터 행동 노드를 검색합니다.
        """
        with self.driver.session() as session:
            return session.execute_read(self._search_behavior_by_angles, skeleton_data)

    @staticmethod
    def _search_behavior_by_angles(tx, skeleton_data):
        """
        각 관절의 각도를 기반으로 행동을 검색합니다.
        가우시안 분포를 사용하여 유사도를 계산합니다.
        """
        results = []
        for joint, data in skeleton_data.items():
            angle = data["각도"] if isinstance(data, dict) else data
            
            query = """
            MATCH (b:Behavior)
            WHERE b[$joint_name + '_mu'] IS NOT NULL
            WITH b, 
                 b[$joint_name + '_mu'] as mu,
                 b[$joint_name + '_sigma'] as sigma,
                 b.name as behavior,
                 b.direction as direction
            WHERE exp(-((($angle - mu) * ($angle - mu)) / (2.0 * sigma * sigma))) >= $threshold
            RETURN DISTINCT behavior, direction,
                   exp(-((($angle - mu) * ($angle - mu)) / (2.0 * sigma * sigma))) as similarity
            ORDER BY similarity DESC
            """
            
            result = tx.run(query, 
                          joint_name=joint,
                          angle=angle,
                          threshold=GAUSSIAN_CONFIG["SIMILARITY_THRESHOLD"])
            
            # 유사도 기반으로 결과 저장
            for record in result:
                results.append({
                    "behavior": record["behavior"],
                    "direction": record["direction"],
                    "similarity": record["similarity"],
                    "joint": joint
                })
        
        # 유사도 기반으로 결과 정렬 및 중복 제거
        results.sort(key=lambda x: x["similarity"], reverse=True)
        unique_results = []
        seen = set()
        for r in results:
            key = f"{r['behavior']}_{r['direction']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results

    def get_behavior_details(self, behavior_name, direction):
        """
        특정 행동 노드의 모든 데이터를 조회합니다.
        """
        with self.driver.session() as session:
            return session.execute_read(self._get_behavior_details, behavior_name, direction)

    @staticmethod
    def _get_behavior_details(tx, behavior_name, direction):
        """
        행동 노드의 상세 정보를 조회하는 Cypher 쿼리를 실행합니다.
        """
        query = """
        MATCH (b:Behavior)
        WHERE b.name = $name AND b.direction = $direction
        RETURN {
            behavior: b.name,
            direction: b.direction,
            skeleton_data: {
                angles: [key IN keys(b) WHERE key ENDS WITH '_angles' | {
                    joint: replace(key, '_angles', ''),
                    angles: b[key]
                }],
                weights: [key IN keys(b) WHERE key ENDS WITH '_weights' | {
                    joint: replace(key, '_weights', ''),
                    weights: b[key]
                }],
                ranges: [key IN keys(b) WHERE key ENDS WITH '_range' | {
                    joint: replace(key, '_range', ''),
                    range: b[key]
                }],
                mu: [key IN keys(b) WHERE key ENDS WITH '_mu' | {
                    joint: replace(key, '_mu', ''),
                    value: b[key]
                }],
                sigma: [key IN keys(b) WHERE key ENDS WITH '_sigma' | {
                    joint: replace(key, '_sigma', ''),
                    value: b[key]
                }]
            },
            descriptions: b.descriptions,
            image_urls: b.image_urls,
            timestamps: b.timestamps
        } as details
        """
        
        result = tx.run(query, name=behavior_name, direction=direction)
        record = result.single()
        return record["details"] if record else None
