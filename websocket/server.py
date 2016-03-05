import sys

from twisted.internet import reactor
from twisted.python import log
from pprint import pprint
from twisted.application.internet import TCPServer
from twisted.application.service import Application
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.resource import Resource
import numpy as np
import scipy as sp
from scipy import stats
import argparse
import json
import struct
from datacube import Datacube

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource


class DatacubeProtocol(WebSocketServerProtocol):

    def __init__(self):
        # This is a dispatch table of functions that respond to the call request
        self.dispatch = {   'cube': self.cube,
                            'jp2': self.jp2,
                            'corr': self.corr
        }

    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def onMessage(self, payload, isBinary):
        print(np.max(data))
        # parse the request
        request = json.loads(payload.decode('utf8'))
        # dispatch to the function
        self.dispatch[request['call']](request)

    # dispatch functions

    # send cube data
    def cube(self, request):
        subscripts = [slice(None,None,None)]*data.ndim
        if 'select' in request:
            assert(isinstance(request['select'], list))
            assert(len(request['select']) == data.ndim)
            for axis, selector in enumerate(request['select']):
                if isinstance(selector, list):
                    if isinstance(selector[0], int):
                        subscripts[axis] = np.array(selector, dtype=np.int)
                    elif isintance(selector[0], bool):
                        subscripts[axis] = np.array(selector, dtype=np.bool)
                elif isinstance(selector, dict):
                    if 'step' in selector:
                        subscripts[axis] = slice(selector['start'], selector['stop'], selector['step'])
                    else:
                        subscripts[axis] = slice(selector['start'], selector['stop'], None)

        shape = data[subscripts].shape
        msg = struct.pack('>I', shape[1],) + struct.pack('>I', shape[0]) + zscore[subscripts].tobytes()
        self.sendMessage(msg, isBinary=True)


    def jp2(self, request):
        self.sendMessage(data[1].tobytes(), isBinary=True)

    def corr(self, request):
        r=datacube.get_correlated(request['seed'], 0)
        print(np.nanmean(r))
        self.sendMessage(json.dumps(np.argsort(-r).tolist()))


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
    zscore = sp.stats.mstats.zscore(data, axis=1)
    print("data ", data[0])

    datacube = Datacube(data)

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
