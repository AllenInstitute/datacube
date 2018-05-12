


class ModelLoader():
    def __init__(self, config):
        self.config = config

    def get (self, id = None):

        if id == None:
            return "No id provided"


        path = self.config.get_property("ccf_model_path")
        path = path.replace("{0}", str(id))

        with open(path) as f:
            out = f.read()

        return out
