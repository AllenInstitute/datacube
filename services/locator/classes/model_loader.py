


class ModelLoader():
    def __init__(self, config):
        self.config = config

    def get (self, id = None):

        if id == None:
            return "No id provided"


        path = self.config.get_property("ccf_model_path")
        path = path.replace("{0}", str(id))

        out = ""

        with open(path) as f:
            for line in f:
                if '#' in line:
                    continue

                out += line

        return out