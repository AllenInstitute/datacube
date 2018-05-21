import txaio
import logging
from twisted.internet import reactor, threads, protocol
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.task import LoopingCall
from twisted.python.threadpool import ThreadPool
#import txpool
import sys
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp.types import RegisterOptions, CallOptions
from autobahn.wamp.auth import compute_wcs
#from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from wamp import ApplicationSession, ApplicationRunner # copy of stock wamp.py with modified timeouts
import numpy as np
import xarray as xr
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
import functools

from six import text_type as str
from six import iteritems
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

        def _ensure_computed(f, *args, **kwargs):
            res = f(*args, **kwargs)
            if isinstance(res, (xr.Dataset, xr.DataArray)):
                return res.compute()
            else:
                return res


        @inlineCallbacks
        def info(name=None):
            try:
                datacube = datacubes[name]
                res = yield threads.deferToThread(_ensure_computed, datacube.info)
                returnValue(res)
            except Exception as e:
                print({'name': name})
                _application_error(e)


        @inlineCallbacks
        def raw(fields=None, select=None, coords=None, filters=None, name=None):
            try:
                datacube = datacubes[name]
                res = yield threads.deferToThread(_ensure_computed, datacube.raw, select, coords, fields, filters)
                returnValue(res.to_dict())
            except Exception as e:
                print({'fields': fields, 'select': select, 'coords': coords, 'name': name, 'filters': filters})
                _application_error(e)


        @inlineCallbacks
        def image(field, select=None, coords=None, image_format='jpeg', dim_order=None, name=None):
            try:
                datacube = datacubes[name]
                res = yield threads.deferToThread(_ensure_computed, datacube.image, select, coords, field, dim_order, image_format)
                returnValue(res)
            except Exception as e:
                print({'field': field, 'select': select, 'dim_order': dim_order, 'image_format': image_format, 'name': name})
                _application_error(e)


        @inlineCallbacks
        def corr(field, dim, seed_idx, fields=None, select=None, coords=None, filters=None, name=None):
            try:
                datacube = datacubes[name]
                res = yield threads.deferToThread(_ensure_computed, datacube.corr, field, dim, seed_idx, select=select, coords=coords, filters=filters)
                res = res.where(res.corr.notnull(), drop=True)
                res = res.sortby('corr', ascending=False)
                if fields is not None:
                    additional_fields_result = yield threads.deferToThread(_ensure_computed, datacube.raw, coords={dim: res[dim].values.tolist()}, fields=fields)
                    res = xr.merge([res, additional_fields_result])
                returnValue(res.to_dict())
            except Exception as e:
                print({'field': field, 'dim': dim, 'seed_idx': seed_idx, 'fields': fields, 'select': select, 'filters': filters, 'name': name})
                _application_error(e)


        @inlineCallbacks
        def conn_spatial_search(voxel, fields=None, select=None, coords=None, filters=None):
            try:
                name = 'connectivity'
                conn_datacube = datacubes[name]
                conn = conn_datacube.df
                # round input voxel to nearest coordinate (conn datacube and streamlines are both at 100 micron resolution)
                voxel['anterior_posterior'] = int(conn.anterior_posterior.sel(anterior_posterior=voxel['anterior_posterior'], method='nearest'))
                voxel['superior_inferior'] = int(conn.superior_inferior.sel(superior_inferior=voxel['superior_inferior'], method='nearest'))
                voxel['left_right'] = int(conn.left_right.sel(left_right=voxel['left_right'], method='nearest'))
                voxel_xyz = [voxel['anterior_posterior'], voxel['superior_inferior'], voxel['left_right']]
                projection_map_dir = args.projection_map_dir

                # timeout should be possible with options=CallOptions(timeout=XYZ), but this won't work until
                #   https://github.com/crossbario/crossbar/issues/299 is implemented.
                # relying on twisted instead.
                d = self.call(u'org.brain_map.locator.get_streamlines_at_voxel', voxel=voxel_xyz, map_dir=projection_map_dir)
                d.addTimeout(10, reactor)
                res = yield d

                streamlines_list = res['results']
                experiment_ids = np.array([e['data_set_id'] for e in streamlines_list])
                coords = coords or {}
                if 'experiment' in coords:
                    coords['experiment'] = np.intersect1d(experiment_ids, coords['experiment'])
                else:
                    coords['experiment'] = experiment_ids
                coords['experiment'] = np.intersect1d(conn.experiment.values, coords['experiment']) #todo: shouldn't need this if the data lines up
                coords['experiment'] = coords['experiment'].tolist()
                res = yield threads.deferToThread(_ensure_computed, conn_datacube.raw, select=select, coords=coords, fields=fields, filters=filters)
                streamlines = xr.Dataset({'streamline': (['experiment'], streamlines_list), 'experiment': experiment_ids})
                res = xr.merge([res, streamlines], join='left')
                returnValue(res.to_dict())
            except Exception as e:
                print({'voxel': voxel, 'fields': fields, 'select': select, 'coords': coords, 'filters': filters})
                _application_error(e)


        def filter_cell_specimens(filters=None,
                                  sort=None,
                                  ascending=None,
                                  start=0,
                                  stop=None,
                                  indexes=None,
                                  fields=None):
            #todo: optimize xarray single-row access, and remove this
            if filters is None and sort is None and (start is None or start==0) and stop is None and isinstance(indexes, list):
                sa=np.load(npy_file, mmap_mode='r')

                def _format_structured_array_response(sa):
                    data = []
                    for field in sa.dtype.names:
                        col = sa[field]
                        # ensure network byte order
                        col = col.astype(col.dtype.str.replace('<', '>').replace('=', '>'))
                        data.append(col.tobytes())
                    data = b''.join(data)
                    return {'num_rows': sa.size,
                            'col_names': [str(name) for name in sa.dtype.names],
                            'col_types': [str(sa[name].dtype.name).replace('bytes', 'string') for name in sa.dtype.names],
                            'item_sizes': [sa[name].dtype.itemsize for name in sa.dtype.names],
                            'data': bytes(zlib.compress(data))}

                ret = np.copy(sa[indexes])
                if fields:
                    ret = ret[fields]
                return _format_structured_array_response(ret)
            else:
                return select('cell_specimens', filters, sort, ascending, start, stop, indexes, fields)


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
                datacube = datacubes[name]
                request_cache_key = json.dumps(['request', name, filters, sort, ascending, start, stop, indexes, fields])
                cached = yield redis.get(request_cache_key)
                if not cached:
                    #res = yield threads.deferToThreadPool(reactor,
                    #                                      thread_pool,
                    res = yield threads.deferToThread(
                                                          _ensure_computed,
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
                    returnValue({'filtered_total': int(res[1]), 'indexes': res[0].tolist()})
                else:
                    ret = yield threads.deferToThread(_format_xr_dataset_response, res)
                    returnValue(ret)
            except Exception as e:
                print({'name': name, 'filters': filters, 'sort': sort, 'ascending': ascending, 'start': start, 'stop': stop, 'indexes': indexes, 'fields': fields})
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
                    'col_types': [str(x[name].dtype.name).replace('str', 'string').replace('bytes', 'string') for name in x.keys()],
                    'item_sizes': [x[name].dtype.itemsize for name in x.keys()],
                    'data': bytes(zlib.compress(data))}


        def _application_error(e):
            traceback.print_exc()
            raise ApplicationError(str('org.brain-map.api.datacube.application_error'), str(e) + '\n' + traceback.format_exc())


        try:
            #todo: not working for some reason
            #thread_pool = ThreadPool()

            #clientCreator = protocol.ClientCreator(reactor, RedisClient)
            #txredis_client = yield clientCreator.connectTCP('localhost', 6379)
            redis = yield txredisapi.ConnectionPool()

            #yield pool.on_ready(timeout=30)

            #def _get_datacube(name=None):
            #    if name is None:
            #        if 1==len(datacubes):
            #            name = list(datacubes.keys())[0]
            #        else:
            #            raise RuntimeError('Must specify datacube name when server has more than one datacube loaded (' + ', '.join(datacubes.keys()) + ').')
            #    return datacubes[name]
            if 'connectivity' in datacubes:
                yield self.register(conn_spatial_search,
                                    u'org.brain-map.api.datacube.conn_spatial_search',
                                    options=RegisterOptions(invoke=u'roundrobin'))
            for name in datacubes.keys():
                yield self.register(lambda: True,
                                    u'org.brain-map.api.datacube.status.' + name + '.' + str(details.session),
                                    options=RegisterOptions())
                yield self.register(functools.partial(info, name=name),
                                    u'org.brain-map.api.datacube.info.' + name,
                                    options=RegisterOptions(invoke=u'roundrobin'))
                yield self.register(functools.partial(raw, name=name),
                                    u'org.brain-map.api.datacube.raw.' + name,
                                    options=RegisterOptions(invoke=u'roundrobin'))
                yield self.register(functools.partial(image, name=name),
                                    u'org.brain-map.api.datacube.image.' + name,
                                    options=RegisterOptions(invoke=u'roundrobin'))
                yield self.register(functools.partial(corr, name=name),
                                    u'org.brain-map.api.datacube.corr.' + name,
                                    options=RegisterOptions(invoke=u'roundrobin'))
                yield self.register(functools.partial(select, name=name),
                                    u'org.brain-map.api.datacube.select.' + name,
                                    options=RegisterOptions(invoke=u'roundrobin'))

            # legacy
            if 'cell_specimens' in datacubes:
                yield self.register(filter_cell_specimens,
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

        print('Server ready. ({})'.format(','.join(datacubes.keys())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datacube Service')
    parser.add_argument('router', help='url of WAMP router to connect to e.g. ws://localhost:9000/ws')
    parser.add_argument('realm', help='WAMP realm name to join')
    parser.add_argument('username', help='WAMP-CRA username')
    parser.add_argument('password', help='WAMP-CRA secret')
    parser.add_argument('dataset_manifest', help='JSON dataset manifest')
    parser.add_argument('--max-records', default=1000, help='maximum records to serve in a single request (default: %(default)s)')
    parser.add_argument('--projection-map-dir', help='path to root of projection map directory structure e.g. /allen/aibs/informatics/heatmap/mouseconn_projection_maps_2017_09_11/P56/')
    args = parser.parse_args()
    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    #todo: get logging from processes working
    #txpool.pool.WorkerProtocol.MAX_LENGTH = sys.maxsize
    #process_pool = txpool.Pool(size=4, log=logging, init_call='datacube.worker.instance.load', init_args=(args.data_dir + 'cell_specimens.npy',))

    npy_file = ''

    datacubes={}
    basepath = os.path.dirname(args.dataset_manifest)
    with open(args.dataset_manifest, 'r') as datasets_json:
        datasets = json.load(datasets_json)
        for dataset in datasets:
            if dataset['enabled']:
                data_dir = os.path.join(os.path.dirname(args.dataset_manifest), dataset['data-dir'])

                if dataset['name'] == 'cell_specimens':
                    npy_file = os.path.join(data_dir, 'cell_specimens.npy')

                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                existing = [os.path.exists(os.path.join(data_dir, f['path'])) for f in dataset['files']]
                if sum(existing) < len(dataset['files']):
                    raise RuntimeError('Refusing to run with ' + str(sum(existing)) + ' files when expecting ' + str(len(dataset['files'])) + ', for dataset "' + dataset['name'] + '". Specify --recache option to generate files (will overwrite existing files).')
                    exit(1)
                data_file = next(f for f in dataset['files'] if re.search('(\.nc|\.zarr\.lmdb)$', f['path']))
                option_keys = ['chunks', 'max_response_size', 'max_cacheable_bytes', 'missing_data', 'calculate_stats', 'persist']
                options = {k:v for k,v in iteritems(data_file) if k in option_keys}
                if not data_file['use_chunks']:
                    del options['chunks']

                datacubes[dataset['name']] = Datacube(
                    dataset['name'],
                    os.path.join(data_dir, data_file['path']),
                    **options)

    runner = ApplicationRunner(str(args.router), str(args.realm))
    runner.run(PandasServiceComponent, auto_reconnect=True)
