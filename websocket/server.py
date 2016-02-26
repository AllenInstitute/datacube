import sys

from twisted.internet import reactor
from twisted.python import log
from twisted.web.server import Site
from twisted.web.static import File
import numpy as np
import pgmagick
import argparse

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource


class DatacubeProtocol(WebSocketServerProtocol):

    def __init__(self):
        # This is a dispatch table of functions that respond to the call request
        self.dispatch = {   'cube': self.cube,
                            'jp2': self.jp2
        }

        # This is the argument parser for all the functions
        self.parser = argparse.ArgumentParser(prog='Datacube')
        self.parser.add_argument('call', nargs='?')

    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def onMessage(self, payload, isBinary):
        # parse the request
        request = self.parser.parse_args(payload.decode('utf8').split())
        # dispatch to the function
        self.dispatch[request.call](request)

    # dispatch functions

    # send cube data
    def cube(self, request):
        self.sendMessage(data[1].tobytes(), isBinary=True)


    def jp2(self, request):
        self.sendMessage(data[1].tobytes(), isBinary=True)

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
