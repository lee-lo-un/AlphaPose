from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node_if_not_exists(self, label, properties):
        """
        노드가 존재하지 않을 경우 새로 생성합니다.
        """
        with self.driver.session() as session:
            session.execute_write(self._create_node_if_not_exists, label, properties)

    @staticmethod
    def _create_node_if_not_exists(tx, label, properties):
        """
        Cypher 쿼리를 실행하여 노드를 MERGE합니다.
        """
        query = f"""
        MERGE (n:{label} {{name: $name}})
        ON CREATE SET n += $properties
        """
        tx.run(query, name=properties["name"], properties=properties)

    def create_relationship(self, label1, name1, label2, name2, relationship_type):
        """
        두 노드 간의 관계를 생성합니다.
        """
        with self.driver.session() as session:
            session.execute_write(self._create_relationship, label1, name1, label2, name2, relationship_type)

    @staticmethod
    def _create_relationship(tx, label1, name1, label2, name2, relationship_type):
        """
        Cypher 쿼리를 실행하여 두 노드 간의 관계를 MERGE합니다.
        """
        query = f"""
        MATCH (a:{label1} {{name: $name1}})
        MATCH (b:{label2} {{name: $name2}})
        MERGE (a)-[:{relationship_type}]->(b)
        """
        tx.run(query, name1=name1, name2=name2)

    def create_body_structure(self):
        """
        인체 구조를 정의하는 노드와 관계를 생성합니다.
        """
        body_parts = [
            {"name": "Body", "angle": 0},
            {"name": "Head", "angle": 0},
            {"name": "LeftShoulder", "angle": 0},
            {"name": "RightShoulder", "angle": 0},
            {"name": "LeftElbow", "angle": 0},
            {"name": "RightElbow", "angle": 0},
            {"name": "LeftHand", "angle": 0},
            {"name": "RightHand", "angle": 0},
            {"name": "LeftHip", "angle": 0},
            {"name": "RightHip", "angle": 0},
            {"name": "LeftKnee", "angle": 0},
            {"name": "RightKnee", "angle": 0},
            {"name": "LeftFoot", "angle": 0},
            {"name": "RightFoot", "angle": 0},
        ]

        # 노드 생성
        for part in body_parts:
            self.create_node_if_not_exists("BodyPart", part)

        # 관계 생성
        relationships = [
            ("Body", "Head"),
            ("Body", "LeftShoulder"),
            ("Body", "RightShoulder"),
            ("LeftShoulder", "LeftElbow"),
            ("RightShoulder", "RightElbow"),
            ("LeftElbow", "LeftHand"),
            ("RightElbow", "RightHand"),
            ("Body", "LeftHip"),
            ("Body", "RightHip"),
            ("LeftHip", "LeftKnee"),
            ("RightHip", "RightKnee"),
            ("LeftKnee", "LeftFoot"),
            ("RightKnee", "RightFoot"),
        ]

        for start, end in relationships:
            self.create_relationship("BodyPart", start, "BodyPart", end, "CONNECTED_TO")

    def query_body_parts(self):
        """
        모든 BodyPart 노드를 쿼리하여 출력합니다.
        """
        with self.driver.session() as session:
            result = session.run("MATCH (n:BodyPart) RETURN n.name AS name, n.angle AS angle")
            for record in result:
                print(f"Name: {record['name']}, Angle: {record['angle']}")

    def query_structure(self):
        """
        인체 구조를 쿼리하여 출력합니다.
        """
        with self.driver.session() as session:
            result = session.run("""
            MATCH (a:BodyPart)-[r:CONNECTED_TO]->(b:BodyPart)
            RETURN a.name AS from, b.name AS to
            """)
            for record in result:
                print(f"{record['from']} -> {record['to']}")

# Neo4j 연결 정보
uri = "neo4j+s://<your-instance-id>.databases.neo4j.io"
user = "neo4j"
password = "<your-password>"

# 클라이언트 생성 및 실행
kg = KnowledgeGraph(uri, user, password)

# 인체 구조 생성
print("Creating body structure...")
kg.create_body_structure()

# 노드 출력
print("\nQuerying body parts...")
kg.query_body_parts()

# 구조 출력
print("\nQuerying body structure...")
kg.query_structure()

# 연결 종료
kg.close()
