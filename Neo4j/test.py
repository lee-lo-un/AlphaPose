from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} {{"
        query += ", ".join([f"{key}: ${key}" for key in properties.keys()])
        query += "})"
        tx.run(query, **properties)

    def create_relationship(self, node1_label, node1_properties, node2_label, node2_properties, relationship_type):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, node1_label, node1_properties, node2_label, node2_properties, relationship_type)

    @staticmethod
    def _create_relationship(tx, node1_label, node1_properties, node2_label, node2_properties, relationship_type):
        query = (
            f"MATCH (a:{node1_label} {{"
            + ", ".join([f"{key}: ${key}" for key in node1_properties.keys()])
            + "}), (b:{node2_label} {{"
            + ", ".join([f"{key}: ${key}" for key in node2_properties.keys()])
            + f"}}) CREATE (a)-[:{relationship_type}]->(b)"
        )
        tx.run(query, **node1_properties, **node2_properties)

# Example usage
kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")

# Create nodes
kg.create_node("BodyPart", {"name": "LeftLeg", "angle": 45})
kg.create_node("BodyPart", {"name": "RightLeg", "angle": 50})

# Create relationship
kg.create_relationship("BodyPart", {"name": "LeftLeg"}, "BodyPart", {"name": "RightLeg"}, "CONNECTED")

kg.close()