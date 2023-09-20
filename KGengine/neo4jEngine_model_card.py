from neo4j import GraphDatabase
import json


class neo4jEngineModelCard:

    def __init__(self, uri='bolt://localhost:7687', user='neo4j', password='admin'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_node(self, data):
        with self.driver.session() as session:
            session.execute_write(self.create_node_from_json_model_card, data)

        self.close()

    def close(self):
        self.driver.close()


    def create_node_from_json_model_card(self, tx, data, leaf=None):
        if leaf is not None:
            if isinstance(data, dict):
                for key, value in data.items():
                    #check value typ for any nested section
                    if isinstance(value, str) or isinstance(value, int) \
                            or isinstance(value, float) or isinstance(value, bool):
                        #find the key value pair the value is not dict
                        if value == False:
                            value = False
                        elif value == True:
                            value = True

                        tx.run(f"MATCH (n) WHERE id(n) = {leaf.element_id} SET n.{key} = $value", value=value)
                    elif isinstance(value, list):
                        check_pure_list = True
                        #check the list, if it contains the dict, then run create function with relationship
                        for item in value:
                            if isinstance(item, dict):
                                check_pure_list = False
                                #create node for each section
                                node = tx.run(f"CREATE (n:{key}) RETURN n").single()[0]

                                tx.run(
                                    f"MATCH (a) WHERE id(a) = {leaf.element_id} MATCH (b) WHERE id(b) = {node.element_id}"
                                    f" CREATE (a)-[r:{key}]->(b)")
                                self.create_node_from_json_model_card(tx, item, leaf=node)


                            if check_pure_list:
                                tx.run(f"MATCH (n) WHERE id(n) = {leaf.element_id} SET n.{key} = $value", value=value)

                    else:
                        #find the key value pair the value is dict, then create a node for the key
                        #and build the relationship
                        node = tx.run(f"CREATE (n:{key}) RETURN n").single()[0]

                        tx.run(f"MATCH (a) WHERE id(a) = {leaf.element_id} MATCH (b) WHERE id(b) = {node.element_id}"
                               f" CREATE (a)-[r:{key}]->(b)")
                        self.create_node_from_json_model_card(tx, value, leaf=node)


        else:
            #start from begining
            #create
            #data_card_metadata - leaf - connect to data_card_metadata - must first one
            #data - parent - connect to data_card_metadata
            #training data - parent - connect to data_card_metadata
            #testing_data - parent - connect to data_card_metadata
            #provenance - leaf - connect to data_card_metadata
            parent = None
            for key, value in data.items():
                if key == 'model_card_metadata':
                    node = tx.run(f"CREATE (n:{key}) RETURN n").single()[0]
                    #update the data_card_metadata entity
                    self.create_node_from_json_model_card(tx, value, leaf=node)
                    parent = node
                elif key == 'provenance':
                    #the last item, provenace, a list of dict
                    for item in value:
                        # create node for each section
                        node = tx.run(f"CREATE (n:{key}) RETURN n").single()[0]

                        tx.run(
                            f"MATCH (a) WHERE id(a) = {parent.element_id} MATCH (b) WHERE id(b) = {node.element_id}"
                            f" CREATE (a)-[r:{key}]->(b)")
                        self.create_node_from_json_model_card(tx, item, leaf=node)
                else:
                    node = tx.run(f"CREATE (n:{key}) RETURN n").single()[0]
                    #connect to parent node
                    tx.run(f"MATCH (a) WHERE id(a) = {parent.element_id} MATCH (b) WHERE id(b) = {node.element_id}"
                           f" CREATE (a)-[r:{key}]->(b)")
                    self.create_node_from_json_model_card(tx, value, leaf=node)



if __name__ == '__main__':

    file = open('/Users/luoyu/PycharmProjects/DataModelCard/card/model/sample_card_template.json')

    data = json.load(file)

    run = neo4jEngineModelCard()
    run.create_node(data)
