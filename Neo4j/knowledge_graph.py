from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def check_body_structure_exists(self):
        """
        기본 신체 구조가 이미 존재하는지 확인합니다.
        """
        with self.driver.session() as session:
            query = """
            MATCH (n)
            WHERE n:Joint
            RETURN count(n) as count
            """
            result = session.run(query).single()
            return result["count"] > 0

    def create_body_structure(self):
        """
        기본 신체 구조를 생성합니다.
        이미 존재하는 경우 생성하지 않습니다.
        """
        if self.check_body_structure_exists():
            return False
            
        with self.driver.session() as session:
            # 기본 관절 생성
            session.run("""
            CREATE (head:Joint {name: 'Head'})
            CREATE (neck:Joint {name: 'Neck'})
            CREATE (lshoulder:Joint {name: 'LeftShoulder'})
            CREATE (rshoulder:Joint {name: 'RightShoulder'})
            CREATE (lelbow:Joint {name: 'LeftElbow'})
            CREATE (relbow:Joint {name: 'RightElbow'})
            CREATE (lwrist:Joint {name: 'LeftWrist'})
            CREATE (rwrist:Joint {name: 'RightWrist'})
            CREATE (lhip:Joint {name: 'LeftHip'})
            CREATE (rhip:Joint {name: 'RightHip'})
            CREATE (lknee:Joint {name: 'LeftKnee'})
            CREATE (rknee:Joint {name: 'RightKnee'})
            CREATE (lankle:Joint {name: 'LeftAnkle'})
            CREATE (rankle:Joint {name: 'RightAnkle'})
            """)

            # 관절 연결 관계 생성
            session.run("""
            MATCH (head:Joint {name: 'Head'})
            MATCH (neck:Joint {name: 'Neck'})
            MATCH (lshoulder:Joint {name: 'LeftShoulder'})
            MATCH (rshoulder:Joint {name: 'RightShoulder'})
            MATCH (lelbow:Joint {name: 'LeftElbow'})
            MATCH (relbow:Joint {name: 'RightElbow'})
            MATCH (lwrist:Joint {name: 'LeftWrist'})
            MATCH (rwrist:Joint {name: 'RightWrist'})
            MATCH (lhip:Joint {name: 'LeftHip'})
            MATCH (rhip:Joint {name: 'RightHip'})
            MATCH (lknee:Joint {name: 'LeftKnee'})
            MATCH (rknee:Joint {name: 'RightKnee'})
            MATCH (lankle:Joint {name: 'LeftAnkle'})
            MATCH (rankle:Joint {name: 'RightAnkle'})
            
            CREATE (head)-[:CONNECTS_TO]->(neck)
            CREATE (neck)-[:CONNECTS_TO]->(lshoulder)
            CREATE (neck)-[:CONNECTS_TO]->(rshoulder)
            CREATE (lshoulder)-[:CONNECTS_TO]->(lelbow)
            CREATE (rshoulder)-[:CONNECTS_TO]->(relbow)
            CREATE (lelbow)-[:CONNECTS_TO]->(lwrist)
            CREATE (relbow)-[:CONNECTS_TO]->(rwrist)
            CREATE (neck)-[:CONNECTS_TO]->(lhip)
            CREATE (neck)-[:CONNECTS_TO]->(rhip)
            CREATE (lhip)-[:CONNECTS_TO]->(lknee)
            CREATE (rhip)-[:CONNECTS_TO]->(rknee)
            CREATE (lknee)-[:CONNECTS_TO]->(lankle)
            CREATE (rknee)-[:CONNECTS_TO]->(rankle)
            """)
            
        return True

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"""
        MERGE (n:{label} {{name: $name}})
        ON CREATE SET n += $properties
        """
        tx.run(query, name=properties["name"], properties=properties)

    @staticmethod
    def _create_relationship(tx, label1, name1, label2, name2, relationship_type):
        query = f"""
        MATCH (a:{label1} {{name: $name1}})
        MATCH (b:{label2} {{name: $name2}})
        MERGE (a)-[:{relationship_type}]->(b)
        """
        tx.run(query, name1=name1, name2=name2)
