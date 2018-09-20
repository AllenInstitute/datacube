import json

class CcfDataStore(object):

    def __init__(self, path):

        with open(path, 'r') as ontology_file:
            self.structures = json.load(ontology_file)

        self.acronym_id_map = { structure['acronym']: structure['id'] for structure in self.structures }
        self.id_summary_map = { 
            structure['id']: {
                'id': structure['id'],
                'abbreviation': structure['acronym'],
                'name': structure['name'],
                'color': structure['color_hex_triplet']
            } for structure in self.structures
        }