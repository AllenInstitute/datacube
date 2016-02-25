import sys

from twisted.internet import reactor
from twisted.python import log
from twisted.web.server import Site
from twisted.web.static import File
import numpy as np

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource


class DatacubeProtocol(WebSocketServerProtocol):

    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def onMessage(self, payload, isBinary):
        for i in range(10):
             self.sendMessage(data[i].tobytes(), True)


if __name__ == '__main__':

    log.startLogging(sys.stdout)

    # load cell types data
    data = np.load('../data/ivscc.npy')
    print("data ", data[0])

    factory = WebSocketServerFactory(u"ws://127.0.0.1:9000")
    factory.protocol = DatacubeProtocol

    resource = WebSocketResource(factory)

    # we server static files under "/" ..
    root = File(".")

    # and our WebSocket server under "/ws"
    root.putChild(u"ws", resource)

    # both under one Twisted Web Site
    site = Site(root)
    reactor.listenTCP(9000, site)

    reactor.run()
