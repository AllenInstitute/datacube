import txaio
import logging
from twisted.internet import reactor, threads, protocol
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.task import LoopingCall
from twisted.python.threadpool import ThreadPool
import txpool
import sys
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp.types import RegisterOptions
from autobahn.wamp.auth import compute_wcs
#from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from wamp import ApplicationSession, ApplicationRunner # copy of stock wamp.py with modified timeouts
import numpy as np
import traceback
import json
import zlib
import argparse
import pickle
#import redis
import txredisapi

from datacube import Datacube


class PandasServiceComponent(ApplicationSession):


    def onConnect(self):
        self.join(unicode(args.realm), [u'wampcra'], unicode(args.username))


    def onChallenge(self, challenge):
        if challenge.method == u'wampcra':
            signature = compute_wcs(unicode(args.password).encode('utf8'), challenge.extra['challenge'].encode('utf8'))
            return signature.decode('ascii')
        else:
            raise Exception("don't know how to handle authmethod {}".format(challenge.method))

    def test(self):
        return 1

    @inlineCallbacks
    def onJoin(self, details):


        @inlineCallbacks
        def select(filters=None,
                   sort=None,
                   ascending=None,
                   start=0,
                   stop=None,
                   indexes=None,
                   fields=None):
            try:
                request_cache_key = json.dumps(['request', filters, sort, ascending, start, stop, indexes, fields])
                cached = yield redis.get(request_cache_key)
                if not cached:
                    #res = yield threads.deferToThreadPool(reactor,
                    #                                      thread_pool,
                    res = yield threads.deferToThread(
                                                          datacube.select,
                                                          'dim_0',
                                                          filters,
                                                          sort,
                                                          ascending,
                                                          start,
                                                          stop,
                                                          indexes,
                                                          fields,
                                                          options={'max_records': args.max_records})

                    #res = yield pool.apply_async('datacube.select',
                    #                             (filters,
                    #                             sort,
                    #                             ascending,
                    #                             start,
                    #                             stop,
                    #                             indexes,
                    #                             fields),
                    #                             {'options': {'max_records': args.max_records}})
                    yield redis.setnx(request_cache_key, pickle.dumps(res))
                else:
                    res = pickle.loads(cached)
                if fields == "indexes_only":
                    returnValue({'filtered_total': res.size, 'indexes': res.tolist()})
                else:
                    ret = yield threads.deferToThread(_format_xr_dataset_response, res)
                    returnValue(ret)
            except Exception as e:
                print({'filters': filters, 'sort': sort, 'ascending': ascending, 'start': start, 'stop': stop, 'indexes': indexes, 'fields': fields})
                _application_error(e)


        #todo: support arbitrary dims
        def _format_xr_dataset_response(x):
            data = []
            for field in x.keys():
                col = x[field].values
                # ensure network byte order
                col = col.astype(col.dtype.str.replace('<', '>').replace('=', '>'))
                data.append(col.tobytes())
            data = b''.join(data)
            return {'num_rows': x.dims['dim_0'],
                    'col_names': [unicode(name) for name in x.keys()],
                    'col_types': [unicode(x[name].dtype.name) for name in x.keys()],
                    'item_sizes': [x[name].dtype.itemsize for name in x.keys()],
                    'data': bytes(zlib.compress(data))}


        def _application_error(e):
            traceback.print_exc()
            raise RuntimeError(u'org.brain-map.api.datacube.application_error', e.__class__.__name__, e.message, e.args, e.__doc__)


        try:
            #todo: not working for some reason
            #thread_pool = ThreadPool()

            #clientCreator = protocol.ClientCreator(reactor, RedisClient)
            #txredis_client = yield clientCreator.connectTCP('localhost', 6379)
            redis = yield txredisapi.ConnectionPool()

            #yield pool.on_ready(timeout=30)

            #todo: make filename configurable
            datacube = yield threads.deferToThread(Datacube, args.data_dir + 'cell_specimens.npy')

            yield self.register(select,
                                u'org.brain-map.api.datacube.select',
                                options=RegisterOptions(invoke=u'roundrobin'))
            # legacy
            yield self.register(select,
                                u'org.alleninstitute.pandas_service.filter_cell_specimens',
                                options=RegisterOptions(invoke=u'roundrobin'))

            def _print_stats():
                thread_pool = reactor.getThreadPool()
                print('thread pool stats:')
                thread_pool.dumpStats()
            stats_loop = LoopingCall(_print_stats)
            stats_loop.start(10*60.)
        except Exception as e:
            print("could not register procedure: {0}".format(e))

        print('Server ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pandas Service')
    parser.add_argument('router', help='url of WAMP router to connect to e.g. ws://localhost:9000/ws')
    parser.add_argument('realm', help='WAMP realm name to join')
    parser.add_argument('username', help='WAMP-CRA username')
    parser.add_argument('password', help='WAMP-CRA secret')
    parser.add_argument('data_dir', help='load CSV and NPY files from this directory')
    parser.add_argument('--max-records', default=1000, help='maximum records to serve in a single request (default: %(default)s)')
    args = parser.parse_args()

    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    #todo: get logging from processes working
    #txpool.pool.WorkerProtocol.MAX_LENGTH = sys.maxsize
    #process_pool = txpool.Pool(size=4, log=logging, init_call='datacube.worker.instance.load', init_args=(args.data_dir + 'cell_specimens.npy',))

    runner = ApplicationRunner(unicode(args.router), unicode(args.realm))
    runner.run(PandasServiceComponent, auto_reconnect=True)
