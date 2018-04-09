import json
import os

class ConfigurationManager ():
    def __init__(self, path = None):
        file_path = ""

        if path == None:
            file_path = "./env_vars.json"
        else:
            file_path = path
        
        if not os.path.exists(file_path) or not os.path.getsize(file_path) > 0:
            raise IOError("file does not exist")

        with open(file_path) as f:
            self.env_vars = json.load(f)

    def get_property(self, key):
        return self.env_vars.get(key,"")
