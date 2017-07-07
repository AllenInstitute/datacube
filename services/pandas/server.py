import txaio
import logging
from twisted.internet import reactor, threads, protocol
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.task import LoopingCall
from twisted.python.threadpool import ThreadPool
#import txpool
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
import subprocess
import os.path
import re

from six import text_type as str
from builtins import bytes

from datacube import Datacube


class PandasServiceComponent(ApplicationSession):


    def onConnect(self):
        self.join(str(args.realm), [u'wampcra'], str(args.username))


    def onChallenge(self, challenge):
        if challenge.method == u'wampcra':
            signature = compute_wcs(str(args.password).encode('utf8'), challenge.extra['challenge'].encode('utf8'))
            return signature.decode('ascii')
        else:
            raise Exception("don't know how to handle authmethod {}".format(challenge.method))

    def test(self):
        return 1

    @inlineCallbacks
    def onJoin(self, details):


        def _get_datacube(name=None):
            if name is None:
                if 1==len(datacubes):
                    name = list(datacubes.keys())[0]
                else:
                    raise RuntimeError('Must specify datacube name when server has more than one datacube loaded (' + ', '.join(datacubes.keys()) + ').')
            return datacubes[name]
            

        @inlineCallbacks
        def raw(fields, select, name=None):
            try:
                datacube = _get_datacube(name)
                res = yield threads.deferToThread(datacube.raw, select, fields)
                returnValue(res.to_dict())
            except Exception as e:
                print({'fields': fields, 'select': select, 'name': name})
                _application_error(e)


        @inlineCallbacks
        def image(field, select, image_format='jpeg', name=None):
            try:
                datacube = _get_datacube(name)
                res = yield threads.deferToThread(datacube.image, select, field, image_format)
                returnValue(res)
            except Exception as e:
                print({'field': field, 'select': select, 'image_format': image_format, 'name': name})
                _application_error(e)


        #todo: this is the 1-d-only analog of "raw", with filtering; these should be combined
        @inlineCallbacks
        def select(name=None,
                   filters=None,
                   sort=None,
                   ascending=None,
                   start=0,
                   stop=None,
                   indexes=None,
                   fields=None):
            try:
                datacube = _get_datacube(name)
                request_cache_key = json.dumps(['request', name, filters, sort, ascending, start, stop, indexes, fields])
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


        #todo: support arbitrary dims (possibly use xr.Dataset.to_dict())
        def _format_xr_dataset_response(x):
            data = []
            for field in x.keys():
                col = x[field].values
                # ensure network byte order
                col = col.astype(col.dtype.str.replace('<', '>').replace('=', '>'))
                data.append(col.tobytes())
            data = b''.join(data)
            return {'num_rows': x.dims['dim_0'],
                    'col_names': [str(name) for name in x.keys()],
                    'col_types': [str(x[name].dtype.name).replace('bytes', 'string') for name in x.keys()],
                    'item_sizes': [x[name].dtype.itemsize for name in x.keys()],
                    'data': bytes(zlib.compress(data))}


        def _application_error(e):
            traceback.print_exc()
            raise RuntimeError(str('org.brain-map.api.datacube.application_error'), e.__class__.__name__, e.message, e.args, e.__doc__)


        try:
            #todo: not working for some reason
            #thread_pool = ThreadPool()

            #clientCreator = protocol.ClientCreator(reactor, RedisClient)
            #txredis_client = yield clientCreator.connectTCP('localhost', 6379)
            redis = yield txredisapi.ConnectionPool()

            #yield pool.on_ready(timeout=30)

            yield self.register(raw,
                                u'org.brain-map.api.datacube.raw',
                                options=RegisterOptions(invoke=u'roundrobin'))
            yield self.register(image,
                                u'org.brain-map.api.datacube.image',
                                options=RegisterOptions(invoke=u'roundrobin'))
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
    parser.add_argument('dataset_manifest', help='JSON dataset manifest')
    parser.add_argument('--max-records', default=1000, help='maximum records to serve in a single request (default: %(default)s)')
    parser.add_argument('--recache', action='store_true', help='overwrite existing data files')
    args = parser.parse_args()

    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    #todo: get logging from processes working
    #txpool.pool.WorkerProtocol.MAX_LENGTH = sys.maxsize
    #process_pool = txpool.Pool(size=4, log=logging, init_call='datacube.worker.instance.load', init_args=(args.data_dir + 'cell_specimens.npy',))

    datacubes={}
    with open(args.dataset_manifest, 'r') as datasets_json:
        datasets = json.load(datasets_json)
        for dataset in datasets:
            if dataset['enabled']:
                existing = [os.path.isfile(dataset['data-dir'] + f['path']) for f in dataset['files']]
                if args.recache or sum(existing) == 0:
                    print(' '.join([dataset['script']] + dataset['arguments']))
                    subprocess.call([dataset['script']] + dataset['arguments'])
                else:
                    if sum(existing) < len(dataset['files']):
                        raise RuntimeError('Refusing to run with ' + str(sum(existing)) + ' files when expecting ' + str(len(dataset['files'])) + ', for dataset "' + dataset['name'] + '". Specify --recache option to generate files (will overwrite existing files).')
                        exit(1)
                nc_file = next(f for f in dataset['files'] if re.search('\.nc$', f['path']))
                chunks = nc_file['chunks'] if nc_file['use_chunks'] else None
                datacubes[dataset['name']] = Datacube(dataset['data-dir'] + nc_file['path'], chunks=chunks)

    runner = ApplicationRunner(str(args.router), str(args.realm))
    runner.run(PandasServiceComponent, auto_reconnect=True)
