import os
import numpy as np
from .ontology_service import OntologyService



class VoxelLookup():
    def __init__(self, config):
        self.config = config
        self.spacing = np.array([25, 25, 25])

    def get (self, coords, results):
        results.setdefault("message", "this route has been removed in favor of datacube core.")
        return results

        if coords == None or len(coords) < 3:
                results.setdefault("message", "missing or invalid volume location")
                return results

        coords = np.array(coords) / self.spacing
        
        return self.get_volume_annotation(coords, results)

    def get_volume_annotation(self, coord, results):
        annotation_file = self.config.get_property("p56_annotation_file")

        annotation_volume = np.load(annotation_file)

        id = annotation_volume[coord[0], coord[1], coord[2]]

        ontology = OntologyService(self.config)

        s = ontology.get_structure_by_id(id)

        results.setdefault('id', int(id))
        results.setdefault('name', s['safe_name'])
        results.setdefault('abbreviation', s['acronym'])
        results.setdefault('color', s['color_hex_triplet'])
        results['success'] = True

        return results
