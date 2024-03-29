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
import lz4.frame
import base64
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


class DatacubeServiceComponent(ApplicationSession):


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
        def corr(field, dim, seed_idx, fields=None, select=None, coords=None, filters=None, drop=True, name=None):
            try:
                datacube = datacubes[name]
                res = yield threads.deferToThread(_ensure_computed, datacube.corr, field, dim, seed_idx, select=select, coords=coords, filters=filters, drop=drop)
                if fields is not None:
                    dims = (dim,) if isinstance(dim, str) else dim
                    res_coords = {d: res[d].values.tolist() for d in dims}
                    additional_fields_result = yield threads.deferToThread(_ensure_computed, datacube.raw, coords=res_coords, fields=fields)
                    res = xr.merge([res, additional_fields_result])
                if res.corr.ndim==1:
                    res = res.sortby('corr', ascending=False)
                returnValue(res.to_dict())
            except Exception as e:
                print({'field': field, 'dim': dim, 'seed_idx': seed_idx, 'fields': fields, 'select': select, 'filters': filters, 'name': name})
                _application_error(e)


        @inlineCallbacks
        def groupby(field, groupby=None, agg_func='size', select=None, coords=None, filters=None, sort=None, ascending=None, name=None):
            try:
                datacube = datacubes[name]
                res = yield threads.deferToThread(_ensure_computed, datacube.groupby, field, groupby=groupby, agg_func=agg_func, select=select, coords=coords, filters=filters, sort=sort, ascending=ascending)
                returnValue(res.to_dict())
            except Exception as e:
                print({'field': field, 'groupby': groupby, 'agg_func': agg_func, 'select': select, 'filters': filters, 'sort': sort, 'ascending': ascending, 'name': name})
                _application_error(e)


        @inlineCallbacks
        def conn_spatial_search(voxel, fields=None, select=None, coords=None, filters=None):
            try:
                name = 'connectivity'
                conn_datacube = datacubes[name]
                conn = conn_datacube.df
                if 'ccf' not in conn.coords:
                    #TODO: move this to data generation script
                    conn = conn.assign_coords(**{'ccf': [int(''.join(map(str, map(int, [x,y,z])))) for x,y,z in zip(conn.anterior_posterior_flat.values.tolist(), conn.superior_inferior_flat.values.tolist(), conn.left_right_flat.values.tolist())]})
                    conn_datacube.df = conn
                fields = [f for f in fields if f != 'projection']+['projection_flat']
                coords = {d: c for d, c in iteritems(coords) if d not in ['anterior_posterior', 'superior_inferior', 'left_right']}
                # round input voxel to nearest coordinate (conn datacube and streamlines are both at 100 micron resolution)
                spacing = conn_datacube.df.ccf_structure.attrs['spacing']
                voxel['anterior_posterior'] = np.around(voxel['anterior_posterior']/spacing[0], decimals=0)*spacing[0]
                voxel['superior_inferior'] = np.around(voxel['superior_inferior']/spacing[1], decimals=0)*spacing[1]
                voxel['left_right'] = np.around(voxel['left_right']/spacing[2], decimals=0)*spacing[2]
                voxel_xyz = [voxel['anterior_posterior'], voxel['superior_inferior'], voxel['left_right']]
                coords['ccf'] = int(''.join(map(str, map(int, voxel_xyz))))
                coords['ccf'] = coords['ccf'] if coords['ccf'] in conn.ccf.values else []
                projection_map_dir = args.projection_map_dir

                # timeout should be possible with options=CallOptions(timeout=XYZ), but this won't work until
                #   https://github.com/crossbario/crossbar/issues/299 is implemented.
                # relying on twisted instead.
                d = self.call(u'org.brain_map.locator.get_streamlines_at_voxel', voxel=voxel_xyz, map_dir=projection_map_dir, string=True)
                d.addTimeout(10, reactor)
                res = yield d
                res = json.loads(lz4.frame.decompress(base64.b64decode(res)).decode())

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
                res = res.rename({'projection_flat': 'projection'})
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
                    res = yield threads.deferToThread(
                        _ensure_computed,
                        datacube.raw,
                        select={'dim_0': {'start': start, 'stop': stop}},
                        coords={'dim_0': indexes} if indexes else indexes,
                        fields=(['index'] if fields == 'indexes_only' else fields),
                        filters=filters,
                        sort=sort,
                        ascending=ascending
                    )

                    yield redis.setnx(request_cache_key, pickle.dumps(res))
                else:
                    res = pickle.loads(cached)
                if fields == "indexes_only":
                    returnValue({'filtered_total': res.dims['dim_0'], 'indexes': res.index.values.tolist()})
                else:
                    ret = yield threads.deferToThread(_format_xr_dataset_response, res)
                    returnValue(ret)
            except Exception as e:
                print({'name': name, 'filters': filters, 'sort': sort, 'ascending': ascending, 'start': start, 'stop': stop, 'indexes': indexes, 'fields': fields})
                _application_error(e)


        def _format_xr_dataset_response(x):
            data = []
            for field in x.variables:
                # ensure ascii
                if x[field].dtype.kind == 'U':
                    x[field] = x[field].astype('S')
                # ensure network byte order
                col = x[field].values
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
                yield self.register(functools.partial(groupby, name=name),
                                    u'org.brain-map.api.datacube.groupby.' + name,
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
            stats_loop.start(24*60*60.)

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
    parser.add_argument('--session_name', type=str, default=None, help='Human-readable unique identifier for this session.')
    parser.add_argument('--max-records', default=1000, help='maximum records to serve in a single request (default: %(default)s)')
    parser.add_argument('--projection-map-dir', help='path to root of projection map directory structure e.g. /allen/aibs/informatics/heatmap/mouseconn_projection_maps_2017_09_11/P56/')
    args = parser.parse_args()
    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging(level='info')

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
                option_keys = ['chunks', 'max_response_size', 'max_cacheable_bytes', 'missing_data', 'calculate_stats', 'one_sample_nan', 'persist']
                options = {k:v for k,v in iteritems(data_file) if k in option_keys}
                if not data_file['use_chunks']:
                    del options['chunks']

                datacubes[dataset['name']] = Datacube(
                    dataset['name'],
                    os.path.join(data_dir, data_file['path']),
                    session_name=args.session_name,
                    **options)

    runner = ApplicationRunner(str(args.router), str(args.realm))
    runner.run(DatacubeServiceComponent, auto_reconnect=True)
