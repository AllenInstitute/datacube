from twisted.web import server, resource
import requests
from builtins import bytes

class StatusResource(resource.Resource):
    isLeaf = True

    def __init__(self, extra):
        resource.Resource.__init__(self)

    def render_GET(self, request):
        return bytes('asdf', 'utf8')
