#import txaio
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession #, ApplicationRunner
from autobahn.wamp.types import RegisterOptions
#from wamp import ApplicationSession, ApplicationRunner # copy of stock wamp.py with modified timeouts
import pandas as pd
#import xarray as xr
import numpy as np
import json
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
MAX_RECORDS = 100000



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
            #print('_filter_cell_specimens')
            #print(reactor.getThreadPool()._queue.qsize())
            r = self.cell_specimens
            if indexes:
                r = r[indexes]
            if filters:
                r = _dataframe_query(r, filters)
            if fields and type(fields) is list:
                r = r[fields]
            r = r[start:stop]

            if fields == "indexes_only":
                return r[r.dtype.names[0]].tolist()
            else:
                if r.size > MAX_RECORDS:
                    raise ValueError('Requested would return ' + str(r.size) + ' records; please limit request to ' + str(MAX_RECORDS) + ' records.')
                else:
                    return base64.b64encode(zlib.compress(r.tobytes()))


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
            col_data = []
            col_names = []
            col_types = []
            for name in df.columns:
                column = df[name]
                data = np.array(column)

                if data.dtype.kind == 'O':
                    if all(isinstance(x, basestring) or x is np.nan or x is None for x in data):
                        data[data == np.array([None])] = b''
                        data = np.array([x for x in data], dtype=np.str)
                col_data.append(data)
                col_names.append(name)
                col_types.append(str(data.dtype))
            out = np.array([tuple(data[j] for data in col_data) for j in range(len(df.index))],
                          dtype=[(str(col_names[i]), col_types[i]) for i in range(len(col_names))])
            return out


        #def _structured_array_to_dataset(sa):
        #    return xr.Dataset({field: ('dim_0', sa[field]) for field in sa.dtype.names})


        def _load_dataframes(): 
            print('Loading bogus englarged cell metrics dataset ...')
            if not os.path.isfile(NPY_FILE):
                if os.path.isfile(CSV_FILE):
                    cell_specimens_df = pd.read_csv(CSV_FILE)
                else:
                    api = BrainObservatoryApi(base_uri=None)
                    cell_specimens_list = api.get_cell_metrics()
                    cell_specimens_df = pd.DataFrame(cell_specimens_list * 7)
                    del cell_specimens_list
                    cell_specimens_df.to_csv(CSV_FILE)
                cell_specimens_sa = _dataframe_to_structured_array(cell_specimens_df)
                del cell_specimens_df
                np.save(NPY_FILE, cell_specimens_sa)
                del cell_specimens_sa
            copyfile(NPY_FILE, SHM_FILE)
            self.cell_specimens = np.load(SHM_FILE, mmap_mode='r')
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
