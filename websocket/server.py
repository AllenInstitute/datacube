import sys

from twisted.internet import reactor
from twisted.python import log
from pprint import pprint
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.resource import Resource
import numpy as np
import json
import jsonschema
import struct
from datacube import Datacube
from database import Database
from dispatch import Dispatch
import dispatch
import traceback


from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource

ERR_NONE = 0
ERR_UNSPECIFIED = 1
ERR_FUNCTION_NAME = 2
ERR_REQUEST_VALIDATION = 3

# Responds to web socket messages with json payloads
class DatacubeProtocol(WebSocketServerProtocol):
    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def _send_error_message(self, err_dict, request):
        if ('binary' in request) and request['binary']:
            self.sendMessage(struct.pack('!I', err_dict['code']) + json.dumps(err_dict, encoding='utf-8'), True)
        else:
            self.sendMessage(json.dumps({'error': err_dict}, encoding='utf-8'), False)

    def onMessage(self, payload, isBinary):
        try:
            # parse the request
            request = json.loads(payload.decode('utf8'))
            # dispatch to the function
            response = dispatch_instance.call(request)

            if ('binary' in request) and request['binary']:
                if not isinstance(response, str):
                    response = json.dumps(response, encoding='utf-8')
                self.sendMessage(struct.pack('!I', ERR_NONE) + response, True)
            else:
                self.sendMessage(json.dumps(response, encoding='utf-8'), False)
        except dispatch.FunctionNameError as e:
            self._send_error_message({'message': e.message, 'code': ERR_FUNCTION_NAME}, request)
        except jsonschema.ValidationError as e:
            message = 'request'
            if len(e.absolute_path) > 0:
                message += '[' + ']['.join(['\'%s\'' % x if isinstance(x, basestring) else str(x) for x in e.absolute_path]) + ']'
            message += ': ' + e.message
            err_dict = {'code': ERR_REQUEST_VALIDATION, 'message': message}
            self._send_error_message(err_dict, request)
        except Exception as e:
            traceback.print_exc()
            err_dict = {'args': e.args, 'class': e.__class__.__name__, 'doc': e.__doc__, 'message': e.message, 'traceback': traceback.format_exc(), 'code': ERR_UNSPECIFIED}
            self._send_error_message(err_dict, request)

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
        return dispatch_instance.functions[msg['call']](msg)

if __name__ == '__main__':
    log.startLogging(sys.stdout)

    # load cell types data, database and dispatch
    data = np.load('../data/ivscc.npy').astype(np.float32)
    datacube = Datacube(data)
    database = Database('postgresql+pg8000://postgres:postgres@ibs-andys-ux3:5432/wh')
    dispatch_instance = Dispatch(datacube, database)

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
