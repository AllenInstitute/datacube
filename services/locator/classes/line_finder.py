import requests
import json

from streamline_loader import StreamlineLoader
from neuron_loader import NeuronLoader

class LineFinder():
    def __init__(self, config, timeout = 100):
        self.config = config
        self.swc_type_id = 303941301
        self.streamline_lookup_path = config.get_property('streamline_lookup')
        self.reconstruction_lookup_path = config.get_property('reconstruction_lookup')
        self.timeout = timeout


    def get(self, id = None, results = None):
        if id == None:
            results.setdefault("message", "id missing")

        # Is it a streamline?
        file_location = self.get_streamline_dir(id)

        if file_location != None:
            try:
                # Go get the streamline and convert that ish
                results.setdefault("type", "streamline")
                loader = StreamlineLoader(file_location)

                results.setdefault("lines", loader.load_lines())
                results.setdefault("injection_sites", loader.load_injections())
                results['success'] = True

            except Exception as e:
                results.setdefault("message", "an error occured while parsing the streamline files: " + e.message)    

            return results
        
        file_location = self.get_neuron_dir(id)
        
        if file_location != None:
            try:
                # Go get the reconstruction and convert that ish
                results.setdefault("type", "reconstruction")
                loader = NeuronLoader(file_location)
                
                res = loader.load()
                
                results.setdefault('lines', res)
                results['success'] = True

            except Exception as e:
                results.setdefault("message", "an error occured while parsing the neuron reconstruction: " + e.message)
                return results

        results.setdefault("message", "invalid streamline or reconstruction id")

        return results

    def get_streamline_dir(self, id):

        self.streamline_lookup_path = self.streamline_lookup_path.format(str(id))

        response = requests.get(self.streamline_lookup_path, timeout = self.timeout)

        if response.status_code != 200:
            return None

        response = response.json()

        if response["num_rows"] == 0:
            return None

        return response["msg"][0]["storage_directory"]


    def get_neuron_dir(self, id):
        self.reconstruction_lookup_path = self.reconstruction_lookup_path.format(str(id))

        response = requests.get(self.reconstruction_lookup_path, timeout = self.timeout)

        if response.status_code != 200:
            return None

        response = response.json()

        if response["num_rows"] == 0:
            return None

        response = response["msg"][0]["neuron_reconstructions"][0]["well_known_files"]

        for file in response:
            if file["well_known_file_type_id"] == self.swc_type_id:
                return file["path"]
