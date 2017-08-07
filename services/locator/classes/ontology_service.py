import json
import os

class OntologyService ():
    def __init__(self, config):
        self.config = config
        
        with open(config.get_property("ccf_ontology")) as f:
            self.ontology = json.load(f)

    def get_structure_by_id(self, id):
        for line in self.ontology:
            if line["id"] == str(id):
                return line

    def get_ontology(self):
        return self.ontology
