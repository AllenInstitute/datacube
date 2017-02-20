#import txaio
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession #, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp.types import RegisterOptions
#from wamp import ApplicationSession, ApplicationRunner # copy of stock wamp.py with modified timeouts
import pandas as pd
#import xarray as xr
import numpy as np
import simplejson
import base64
import zlib
import os
import sys
from shutil import copyfile

from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi

DATA_DIR = '../data/' # TODO: pass in through crossbar config
CSV_FILE = DATA_DIR + 'cell_specimens.csv'
NPY_FILE = DATA_DIR + 'cell_specimens.npy'
SHM_FILE = '/dev/shm/cell_specimens.npy'
MAX_RECORDS = 1000



# Responds to wamp rpc
class PandasServiceComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):

        def filter_cell_specimens(filters=None,
                                  start=0,
                                  stop=None,
                                  indexes=None,
                                  fields=None):
            #print('deferToThread')
            d = threads.deferToThread(_filter_cell_specimens, filters, start, stop, indexes, fields)
            return d


        def _filter_cell_specimens(filters=None,
                                  start=0,
                                  stop=None,
                                  indexes=None,
                                  fields=None):
            try:
                #print('_filter_cell_specimens')
                #print(reactor.getThreadPool()._queue.qsize())
                #r = self.cell_specimens
                r = np.load(SHM_FILE, mmap_mode='r')
                if not filters and not fields == "indexes_only":
                    if indexes:
                        num_results = np.array(indexes)[start:stop].size
                    else:
                        num_results = r['index'][start:stop].size
                    if num_results > MAX_RECORDS:
                        raise ValueError('Requested would return ' + str(num_results) + ' records; please limit request to ' + str(MAX_RECORDS) + ' records.')
                if indexes:
                    r = r[indexes]
                if filters:
                    r = _dataframe_query(r, filters)
                filtered_total = r.size;
                if fields and type(fields) is list:
                    r = r[fields]
                r = r[start:stop]

                if fields == "indexes_only":
                    return {'filtered_total': filtered_total, 'indexes': r['index'].tolist()}
                else:
                    if r.size > MAX_RECORDS:
                        raise ValueError('Requested would return ' + str(r.size) + ' records; please limit request to ' + str(MAX_RECORDS) + ' records.')
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
                    'col_names': list(sa.dtype.names),
                    'col_types': [sa[name].dtype.name for name in sa.dtype.names],
                    'item_sizes': [sa[name].dtype.itemsize for name in sa.dtype.names],
                    'data': base64.b64encode(zlib.compress(data))}


        def _application_error(e):
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
            print('Loading bogus englarged cell metrics dataset ...')
            if not os.path.isfile(NPY_FILE):
                if os.path.isfile(CSV_FILE):
                    cell_specimens_df = pd.read_csv(CSV_FILE, index_col='index')
                else:
                    api = BrainObservatoryApi(base_uri=None)
                    cell_specimens_list = api.get_cell_metrics()
                    cell_specimens_df = pd.DataFrame(cell_specimens_list * 7)
                    del cell_specimens_list
                    cell_specimens_df.to_csv(CSV_FILE, index_label='index')
                cell_specimens_sa = _dataframe_to_structured_array(cell_specimens_df)
                del cell_specimens_df
                np.save(NPY_FILE, cell_specimens_sa)
                del cell_specimens_sa
            copyfile(NPY_FILE, SHM_FILE)
            #self.cell_specimens = np.load(SHM_FILE, mmap_mode='r')
            #self.cell_specimens = _structured_array_to_dataset(cell_specimens_shm)
            print('Done.')

        try:
            yield threads.deferToThread(_load_dataframes)
            yield self.register(filter_cell_specimens,
                                u'org.alleninstitute.pandas_service.filter_cell_specimens',
                                options=RegisterOptions(invoke=u'roundrobin'))
        except Exception as e:
            print("could not register procedure: {0}".format(e))

        print("Server ready.")
