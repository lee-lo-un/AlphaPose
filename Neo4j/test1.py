from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            session.execute_write(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} {{"
        query += ", ".join([f"{key}: ${key}" for key in properties.keys()])
        query += "})"
        tx.run(query, **properties)

    def create_body_structure(self):
        body_parts = [
            {"name": "Body", "angle": 0},
            {"name": "LeftElbow", "angle": 0},
            {"name": "RightElbow", "angle": 0},
            {"name": "LeftHand", "angle": 0},
            {"name": "RightHand", "angle": 0},
            {"name": "LeftKnee", "angle": 0},
            {"name": "RightKnee", "angle": 0},
            {"name": "LeftFoot", "angle": 0},
            {"name": "RightFoot", "angle": 0},
            {"name": "Head", "angle": 0}
        ]

        for part in body_parts:
            self.create_node("BodyPart", part)

    def query_body_parts(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:BodyPart) RETURN n.name AS name, n.angle AS angle")
            for record in result:
                print(f"Name: {record['name']}, Angle: {record['angle']}")

# Neo4j Aura 연결 정보
uri = "neo4j+s://b79e9e00.databases.neo4j.io"
user = "neo4j"
password = "mB-DyJYNRRRH-ZjQUo-ZKHVQA0Hpb-qQVVQKYR0du7c"

# 클라이언트 생성 및 실행
kg = KnowledgeGraph(uri, user, password)

# 노드 생성
kg.create_body_structure()

# 노드 쿼리 및 출력
kg.query_body_parts()

# 연결 종료
kg.close()