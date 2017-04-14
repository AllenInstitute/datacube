from autobahn import wamp

@wamp.error(u"org.alleninstitute.datacube.error.selector")
class SelectorError(RuntimeError):
    pass

@wamp.error(u"org.alleninstitute.datacube.error.function_name")
class FunctionNameError(RuntimeError):
    pass

@wamp.error(u"org.alleninstitute.datacube.error.parallel_execution")
class ParallelError(RuntimeError):
    pass

@wamp.error(u"org.alleninstitute.datacube.error.service")
class ServiceError(RuntimeError):
    pass

@wamp.error(u"org.alleninstitute.datacube.error.json")
class RequestNotValidJSON(RuntimeError):
    pass

@wamp.error(u"org.alleninstitute.datacube.error.datacube_name")
class DatacubeNameError(RuntimeError):
    def __init__(self, cube_name):
        super(DatacubeNameError, self).__init__('Datacube named ''{0}'' does not exist.'.format(cube_name))
        self.cube_name = cube_name

@wamp.error(u"org.alleninstitute.datacube.error.multiple_datacubes")
class DatacubeUnspecified(RuntimeError):
    def __init__(self):
        super(DatacubeUnspecified, self).__init__('Server has multiple datacubes; specify a datacube name using the "cube" parameter.')

@wamp.error(u"org.alleninstitute.datacube.error.datacube_selector")
class DatacubeSelectorError(IndexError):
    pass    

@wamp.error(u"org.alleninstitute.datacube.error.unspecified")
class UnspecifiedError(RuntimeError):
    pass
