import txaio
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
import pandas as pd
import json
import base64
import os
import sys
import zlib

from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi

DATA_DIR = './data/'

CROSSBAR_ROUTER_URL = u"ws://127.0.0.1:8081/ws"
WAMP_REALM_NAME = u"pandas_service"

# Responds to wamp rpc
class PandasServiceComponent(ApplicationSession):
    @inlineCallbacks
    def onJoin(self, details):

        def filter_cell_specimens(filters=None,
                                  start=0,
                                  stop=None,
                                  fields=None):
            r = cell_specimens
            if filters:
                r = dataframe_query(r, filters)
            if fields:
                r = r[fields]
            r = r[start:stop]
            json = r.to_json(orient='split')
            if len(json) > 100e6:
                raise ValueError('Requested data too large (' + str(len(json)) + ' bytes); please make a smaller request.')
            return json

 
        def dataframe_query(df, filters):
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
                        return (df[field].isin(value))

                cond = _apply_op(filters[0]['op'], filters[0]['field'], filters[0]['value'])
                for f in filters[1:]:
                    cond &= _apply_op(f['op'], f['field'], f['value'])

                return df[cond]


        try:
            yield self.register(filter_cell_specimens, u'org.alleninstitute.pandas_service.filter_cell_specimens')
        except Exception as e:
            print("could not register procedure: {0}".format(e))

        print("Server ready.")


if __name__ == '__main__':
    print('Loading bogus englarged cell metrics dataset ...')
    api = BrainObservatoryApi(base_uri=None)
    cell_specimens_list = api.get_cell_metrics()
    cell_specimens = pd.DataFrame(cell_specimens_list * 7)
    print('Done.')

    # logging
    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    # wamp
    runner = ApplicationRunner(CROSSBAR_ROUTER_URL, WAMP_REALM_NAME)
    runner.run(PandasServiceComponent)
