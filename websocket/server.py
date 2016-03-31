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
from database import Database
from dispatch import Dispatch

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource


# Responds to web socket messages with json payloads
class DatacubeProtocol(WebSocketServerProtocol):
    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def onMessage(self, payload, isBinary):
        try:
            # parse the request
            request = json.loads(payload.decode('utf8'))
            # dispatch to the function
            self.sendMessage(dispatch.functions[request['call']](request), ('binary' in request and request['binary'] == True))
        except Exception as e:
            self.sendMessage(e.message, False)

# Api responds to http requests with json msg argument
class Api(Resource):
    def render_GET(self, request):
        return self.process(request)

    def render_POST(self, request):
        return self.process(request)

    def process(self, request):
        # parse the request
        pprint(request.args['msg'])
        msg = json.loads(request.args['msg'][0])
        # dispatch to the function
        return dispatch.functions[msg['call']](msg)

if __name__ == '__main__':
    log.startLogging(sys.stdout)

    # load cell types data, database and dispatch
    data = np.load('../data/ivscc.npy').astype(np.float32)
    datacube = Datacube(data)
    database = Database('postgresql+pg8000://postgres:postgres@ibs-andys-ux3:5432/wh')
    dispatch = Dispatch(datacube, database)

    # Serve static files under "/" ..
    root = File(".")

    # WebSocket server under "/ws"
    factory = WebSocketServerFactory(u"ws://127.0.0.1:9000")
    factory.protocol = DatacubeProtocol
    root.putChild(u"ws", WebSocketResource(factory))

    # respond to tab separated value request under "/api"
    root.putChild(u"api", Api())

    site = Site(root)
    reactor.listenTCP(9000, site)

    reactor.run()
