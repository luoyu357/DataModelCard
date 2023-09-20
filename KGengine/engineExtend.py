from neo4j import GraphDatabase

class neo4jExtend:

    def __init__(self, uri='bolt://localhost:7687', user='neo4j', password='admin'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_node(self, tx):
        tx.run(f"MATCH (n) WHERE id(n) = {leaf.element_id} SET n.{key} = $value", value=value)
