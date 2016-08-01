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
import argparse
import pg8000
import os

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol

from autobahn.twisted.resource import WebSocketResource

DATA_DIR = './data/'

DB_HOST = 'testdb2'
DB_PORT = 5942
DB_NAME = 'warehouse-R193'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'

class RequestNotValidJSON(RuntimeError):
    pass

ERR_NONE = 0
ERR_UNSPECIFIED = 1
ERR_NOT_JSON = 2
ERR_FUNCTION_NAME = 3
ERR_REQUEST_VALIDATION = 4
ERR_MIXED_TYPES_IN_SELECTOR = 5

# Responds to web socket messages with json payloads
class DatacubeProtocol(WebSocketServerProtocol):
    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))

    def _send_error_message(self, err_dict, binary):
        if binary:
            self.sendMessage(struct.pack('!I', err_dict['code']) + json.dumps(err_dict, encoding='utf-8'), True)
        else:
            self.sendMessage(json.dumps({'error': err_dict}, encoding='utf-8'), False)

    def onMessage(self, payload, isBinary):
        try:
            binary = False
            try:
                # parse the request
                request = json.loads(payload.decode('utf8'))
            except ValueError as e:
                raise RequestNotValidJSON(e.message)

            binary = ('binary' in request) and request['binary']

            # dispatch to the function
            response = dispatch_instance.call(request)

            if binary:
                if not isinstance(response, str):
                    response = json.dumps(response, encoding='utf-8')
                self.sendMessage(struct.pack('!I', ERR_NONE) + response, True)
            else:
                self.sendMessage(json.dumps(response, encoding='utf-8'), False)
        except RequestNotValidJSON as e:
            self._send_error_message({'message': e.message, 'code': ERR_NOT_JSON}, binary)
        except dispatch.FunctionNameError as e:
            self._send_error_message({'message': e.message, 'code': ERR_FUNCTION_NAME}, binary)
        except dispatch.MixedTypesInSelector as e:
            self._send_error_message({'message': e.message, 'code': ERR_MIXED_TYPES_IN_SELECTOR, 'axis': e.axis}, binary)
        except jsonschema.ValidationError as e:
            message = 'request'
            if len(e.absolute_path) > 0:
                message += '[' + ']['.join(['\'%s\'' % x if isinstance(x, basestring) else str(x) for x in e.absolute_path]) + ']'
            message += ': ' + e.message
            err_dict = {'code': ERR_REQUEST_VALIDATION, 'message': message}
            self._send_error_message(err_dict, binary)
        except CompositeError as e:
            # error from parallel nodes needs special call to render_traceback()
            # and comes back with ANSI control codes which need to be stripped
            import re
            ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
            tb = '\n'.join([ansi_escape.sub('', x) for x in e.render_traceback()])
            print tb
            # TODO: provide information about the compute nodes where the error(s) happened
            err_dict = {'code': ERR_UNSPECIFIED, 'message': tb}
            self._send_error_message(err_dict, binary)
        except Exception as e:
            traceback.print_exc()
            err_dict = {'args': e.args, 'class': e.__class__.__name__, 'doc': e.__doc__, 'message': e.message, 'traceback': traceback.format_exc(), 'code': ERR_UNSPECIFIED}
            self._send_error_message(err_dict, binary)

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

    conn = pg8000.connect(user=DB_USER, host=DB_HOST, port=DB_PORT, database=DB_NAME, password=DB_PASSWORD)
    cursor = conn.cursor()
    cursor.execute("select name from data_cube_runs")
    results = cursor.fetchall()
    #available_cubes = str(results[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=9000)
    #parser.add_argument('--include-cubes', type=str)
    args = parser.parse_args()
    #args.include_cubes = args.include_cubes.split(',')

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    data = None
    if not os.path.exists(DATA_DIR + 'cell_types.npy'):
        print 'Loading cell_types data cube from warehouse ...'
        cursor.execute("select data from data_cubes dc join data_cube_runs dcr on dc.data_cube_run_id = dcr.id where dcr.name = '%s'" % 'cell_types')
        results = cursor.fetchall()
        data = np.asarray(results, dtype=np.float32)[:,0]
        np.save(DATA_DIR + 'cell_types.npy', data)
    else:
        print 'Loading cell_types data cube from filesystem ...'
        data = np.load(DATA_DIR + 'cell_types.npy').astype(np.float32)

    if args.distributed:
        from ipyparallel.error import CompositeError
    else:
        class CompositeError(RuntimeError):
            pass

    # load cell types data, database and dispatch
    datacube = Datacube(data, distributed=args.distributed, observed=~np.isnan(data))
    database = Database('postgresql+pg8000://' + DB_USER + ':' + DB_PASSWORD + '@' + DB_HOST + ':' + str(DB_PORT) + '/' + DB_NAME)
    dispatch_instance = Dispatch(datacube, database)

    # Serve static files under "/" ..
    root = File(".")

    # WebSocket server under "/ws"
    factory = WebSocketServerFactory(u"ws://127.0.0.1:"+str(args.port))
    factory.protocol = DatacubeProtocol
    root.putChild(u"ws", WebSocketResource(factory))

    # respond to tab separated value request under "/api"
    root.putChild(u"api", Api())

    site = Site(root)
    reactor.listenTCP(args.port, site)

    reactor.run()
