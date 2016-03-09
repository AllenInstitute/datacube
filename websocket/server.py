import sys

from twisted.internet import reactor
from twisted.python import log
from pprint import pprint
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.resource import Resource
import numpy as np
import json
from datacube import Datacube
from dispatch import Dispatch

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource


# Responds to web socket messages with json payloads
class DatacubeProtocol(WebSocketServerProtocol):
    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def onMessage(self, payload, isBinary):
        # parse the request
        request = json.loads(payload.decode('utf8'))
        # dispatch to the function
        self.sendMessage(dispatch.functions[request['call']](request), dispatch.isBinary[request['call']])


# Responds to https requests with json payloads
class TsvPage(Resource):
    def render_GET(self, request):
        return self.process(request)

    def render_POST(self, request):
        return self.process(request)

    def process(self, request):
        pprint(request.__dict__)
        newdata = request.content.getvalue()
        print newdata
        return ''

if __name__ == '__main__':
    log.startLogging(sys.stdout)

    # load cell types data
    data = np.load('../data/ivscc.npy').astype(np.float32)
    datacube = Datacube(data)
    dispatch = Dispatch(datacube)

    factory = WebSocketServerFactory(u"ws://127.0.0.1:9000")
    factory.protocol = DatacubeProtocol

    resource = WebSocketResource(factory)

    # we server static files under "/" ..
    root = File(".")

    # and our WebSocket server under "/ws"
    root.putChild(u"ws", resource)

    # respond to tab separated value request under "/tsv"
    root.putChild(u"tsv", TsvPage())

    # both under one Twisted Web Site
    site = Site(root)
    reactor.listenTCP(9000, site)

    reactor.run()
