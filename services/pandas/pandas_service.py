import txaio
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp.types import RegisterOptions
#from wamp import ApplicationSession, ApplicationRunner # copy of stock wamp.py with modified timeouts
import pandas as pd
#import xarray as xr
import numpy as np
from scipy.stats import rankdata
#import simplejson
#import base64
import zlib
import os
import sys
import glob
import re
import urllib
import argparse
from shutil import copyfile

#from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi


class PandasServiceComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):

        def filter_cell_specimens(name=None,
                                  filters=None,
                                  sort=None,
                                  ascending=None,
                                  start=0,
                                  stop=None,
                                  indexes=None,
                                  fields=None):
            #print('deferToThread')
            d = threads.deferToThread(_filter_cell_specimens, name, filters, sort, ascending, start, stop, indexes, fields)
            return d


        def _filter_cell_specimens(name=None,
                                   filters=None,
                                   sort=None,
                                   ascending=None,
                                   start=0,
                                   stop=None,
                                   indexes=None,
                                   fields=None):
            try:
                #print('_filter_cell_specimens')
                #print(reactor.getThreadPool()._queue.qsize())
                #r = self.cell_specimens
                if not name and len(self.keys) == 1:
                    name = self.keys[0]
                if name not in self.keys:
                    raise ValueError('Requested name \'' + str(name) + '\' does not exist; choose one of (' + ','.join(self.keys) + ').')
                if args.use_mmap:
                    r = np.load(args.mmap_dir + name + '.npy', mmap_mode='r')
                else:
                    r = self.data[name]
                if not filters and not fields == "indexes_only":
                    if indexes:
                        num_results = np.array(indexes)[start:stop].size
                    else:
                        num_results = r['index'][start:stop].size
                    if num_results > args.max_records:
                        raise ValueError('Requested would return ' + str(num_results) + ' records; please limit request to ' + str(args.max_records) + ' records.')
                if indexes:
                    r = r[indexes]
                if filters:
                    r = _dataframe_query(r, filters)
                filtered_total = r.size;
                if sort:
                    if not ascending:
                        ascending = [True] * len(sort)
                    for field in sort:
                        if field not in r.dtype.names:
                            raise ValueError('Requested sort field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(r.dtype.names))
                    s = r[sort]
                    ranks = np.zeros((s.size, len(s.dtype)))
                    for idx, (field, asc) in enumerate(zip(sort[::-1], ascending[::-1])):
                        if s[field].dtype.name.startswith('string') or s[field].dtype.name.startswith('unicode'):
                            blank_count = np.count_nonzero(s[field] == '')
                        else:
                            blank_count = np.count_nonzero(np.isnan(s[field]))
                        ranks[:,idx] = rankdata(s[field], method='dense')
                        if not asc:
                            ranks[:,idx] = np.roll(-ranks[:,idx], -blank_count)
                    r = r[np.lexsort(tuple(ranks[:,idx] for idx in range(ranks.shape[1])))]
                if fields and type(fields) is list:
                    for field in fields:
                        if field not in r.dtype.names:
                            raise ValueError('Requested field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(r.dtype.names))
                    r = r[fields]
                r = r[start:stop]

                if fields == "indexes_only":
                    return {'filtered_total': filtered_total, 'indexes': r['index'].tolist()}
                else:
                    if r.size > args.max_records:
                        raise ValueError('Requested would return ' + str(r.size) + ' records; please limit request to ' + str(args.max_records) + ' records.')
                    else:
                        #return base64.b64encode(zlib.compress(r.tobytes()))
                        return _format_structured_array_response(r)
            except Exception as e:
                _application_error(e)


        def _format_structured_array_response(sa):
            #records = [{field: sa[i][field].item() for field in sa[i].dtype.names} for i in range(sa.size)]
            #return base64.b64encode(zlib.compress(simplejson.dumps(records, ignore_nan=True)))
            data = []
            for field in sa.dtype.names:
                col = sa[field]
                # ensure network byte order
                col = col.astype(col.dtype.str.replace('<', '>').replace('=', '>'))
                data.append(col.tobytes())
            data = b''.join(data)
            return {'num_rows': sa.size,
                    'col_names': [unicode(name) for name in sa.dtype.names],
                    'col_types': [unicode(sa[name].dtype.name) for name in sa.dtype.names],
                    'item_sizes': [sa[name].dtype.itemsize for name in sa.dtype.names],
                    #'data': base64.b64encode(zlib.compress(data))}
                    'data': bytes(zlib.compress(data))}


        def _application_error(e):
            print(e)
            raise ApplicationError(u'org.alleninstitute.pandas_service.application_error', e.__class__.__name__, e.message, e.args, e.__doc__)
            

        def _dataframe_query(df, filters):
            if not filters:
                return df
            else:
                def _apply_op(op, field, value):
                    if op == '=' or op == 'is':
                        return (df[field] == value)
                    elif op == '<':
                        return (df[field] < value)
                    elif op == '>':
                        return (df[field] > value)
                    elif op == '<=':
                        return (df[field] <= value)
                    elif op == '>=':
                        return (df[field] >= value)
                    elif op == 'between':
                        return ((df[field] >= value[0]) & (df[field] <= value[1]))
                    elif op == 'in':
                        return np.any(df[field][:,np.newaxis] == np.array([value]), axis=1)

                cond = _apply_op(filters[0]['op'], filters[0]['field'], filters[0]['value'])
                for f in filters[1:]:
                    cond &= _apply_op(f['op'], f['field'], f['value'])

                return df[cond]


        def _dataframe_to_structured_array(df):
            if 'index' in list(df):
                col_data = []
                col_names = []
                col_types = []
            else:
                col_data = [df.index]
                col_names = ['index']
                col_types = ['i4']
            for name in df.columns:
                column = df[name]
                data = np.array(column)

                if data.dtype.kind == 'O':
                    if all(isinstance(x, basestring) or x is np.nan or x is None for x in data):
                        data[data == np.array([None])] = b''
                        data[np.array([True if str(x) == 'nan' else False for x in data], dtype=np.bool)] = b''
                        data = np.array([x + '\0' for x in data], dtype=np.str)
                col_data.append(data)
                col_names.append(name)
                # javascript cannot natively handle longs
                if str(data.dtype) == 'int64':
                    col_types.append('i4')
                elif str(data.dtype) == 'uint64':
                    col_types.append('u4')
                else:
                    col_types.append(data.dtype.str)
            out = np.array([tuple(data[j] for data in col_data) for j in range(len(df.index))],
                          dtype=[(str(col_names[i]), col_types[i]) for i in range(len(col_names))])
            return out


        #def _structured_array_to_dataset(sa):
        #    return xr.Dataset({field: ('dim_0', sa[field]) for field in sa.dtype.names})


        def _load_dataframes(): 
            print('Loading ...')
            self.keys = []
            self.data = {}
            for datafile in glob.glob(args.data_dir + '*.csv') + glob.glob(args.data_dir + '*.npy'):
                filename = os.path.basename(datafile)
                
                if not os.path.isfile(args.cache_dir + filename) \
                        or os.path.getmtime(args.cache_dir + filename) < os.path.getmtime(datafile):
                    copyfile(datafile, args.cache_dir + filename)
                    
                    if re.match('.*\.csv$', datafile, re.I):
                        df = pd.read_csv(args.cache_dir + filename)
                        df.to_csv(args.cache_dir + filename, index=('index' not in list(df)), index_label='index')
                        sa = _dataframe_to_structured_array(df)
                        del df
                        np.save(args.cache_dir + re.sub('\.csv', '.npy', filename, flags=re.I), sa)
                        del sa
            for npyfile in glob.glob(args.cache_dir + '*.npy'):
                name = os.path.splitext(os.path.basename(npyfile))[0]
                self.keys.append(name)
                if args.use_mmap:
                    if args.cache_dir != args.mmap_dir:
                        copyfile(npyfile, args.mmap_dir + name + '.npy')
                else:
                    self.data[name] = np.load(npyfile)


        try:
            yield threads.deferToThread(_load_dataframes)
            yield self.register(filter_cell_specimens,
                                u'org.alleninstitute.pandas_service.filter_cell_specimens',
                                options=RegisterOptions(invoke=u'roundrobin'))
        except Exception as e:
            print("could not register procedure: {0}".format(e))

        print('Server ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pandas Service')
    parser.add_argument('router', help='url of WAMP router to connect to e.g. ws://localhost:9000/ws')
    parser.add_argument('realm', help='WAMP realm name to join')
    parser.add_argument('data_dir', help='load CSV and NPY files from this directory')
    parser.add_argument('--cache-dir', default='./data/', help='local data store (default: "%(default)s")')
    parser.add_argument('--mmap-dir', default='/dev/shm/', help='copy files to this directory before memory-mapping them (can be same as --cache-dir to avoid a 2nd copy) (default: "%(default)s")')
    parser.add_argument('--no-mmap', action='store_false', dest='use_mmap', help='don\'t use memory-mapped files; load the data into memory')
    parser.add_argument('--max-records', default=1000, help='maximum records to serve in a single request (default: %(default)s)')
    parser.add_argument('--demo', action='store_true', dest='demo', help='load demo dataset')
    parser.set_defaults(use_mmap=True, demo=False)
    args = parser.parse_args()

    if args.demo:
        csv_url = 'http://api.brain-map.org/api/v2/data/ApiCamCellMetric/query.csv?num_rows=all'
        csv_file = args.data_dir + 'cell_specimens.csv'
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)
        if not os.path.isfile(csv_file):
            urllib.urlretrieve(csv_url, csv_file)

    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    runner = ApplicationRunner(unicode(args.router), unicode(args.realm))
    runner.run(PandasServiceComponent, auto_reconnect=True)