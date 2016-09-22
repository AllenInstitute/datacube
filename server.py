import txaio
from pprint import pprint
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
import numpy as np
import json
import base64
import traceback
import argparse
import pg8000
import os
import sys
from progressbar import ProgressBar, Percentage, Bar, ETA, Counter, FileTransferSpeed

from datacube import Datacube
from database import Database
import cam
import error


DATA_DIR = './data/'

DB_HOST = 'testwarehouse1'
DB_PORT = 5432
DB_NAME = 'warehouse-R193'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'

CROSSBAR_ROUTER_URL = u"ws://127.0.0.1:8080/ws"
WAMP_REALM_NAME = u"datacube"

# Responds to wamp rpc
class DatacubeComponent(ApplicationSession):
    @inlineCallbacks
    def onJoin(self, details):

        def info(cube=None):
            try:
                cube = self._select_cube(cube)
                return {
                    'ndim': datacube[cube].ndim,
                    'shape': list(datacube[cube].shape),
                    'dtype': datacube[cube].dtype.name
                }
            except Exception as e:
                self._format_unspecified_error(e)

        def raw(select, cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].raw(select)
                return self._format_data_response(data)
            except Exception as e:
                self._format_unspecified_error(e)

        def log2_1p(select, cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].log2_1p(select)
                return self._format_data_response(data)
            except Exception as e:
                self._format_unspecified_error(e)

        def standard(select, axis, cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].standard(select, axis)
                return self._format_data_response(data)
            except Exception as e:
                self._format_unspecified_error(e)

        def corr(select, axis, seed, cube=None):
            try:
                cube = self._select_cube(cube)
                return datacube[cube].corr(select, axis, seed)
            except Exception as e:
                self._format_unspecified_error(e)

        def fold_change(numerator, denominator, axis, cube=None):
            try:
                cube = self._select_cube(cube)
                return datacube[cube].fold_change(numerator, denominator, axis)
            except Exception as e:
                self._format_unspecified_error(e)

        def diff(numerator, denominator, axis, cube=None):
            try:
                cube = self._select_cube(cube)
                return datacube[cube].diff(numerator, denominator, axis)
            except Exception as e:
                self._format_unspecified_error(e)

        try:
            yield self.register(raw, u'org.alleninstitute.datacube.raw')
            yield self.register(log2_1p, u'org.alleninstitute.datacube.log2_1p')
            yield self.register(standard, u'org.alleninstitute.datacube.standard')
            yield self.register(info, u'org.alleninstitute.datacube.info')
            yield self.register(corr, u'org.alleninstitute.datacube.corr')
            yield self.register(fold_change, u'org.alleninstitute.datacube.fold_change')
            yield self.register(diff, u'org.alleninstitute.datacube.diff')
        except Exception as e:
            print("could not register procedure: {0}".format(e))

        print("Server ready.")

    def _select_cube(self, cube):
        if cube is not None:
            if cube not in datacube:
                raise error.DatacubeNameError(cube)
        else:
            if 1 == len(datacube):
                cube = datacube.keys()[0]
            else:
                raise error.DatacubeUnspecified()
        return cube

    # Return values from a section of the datacube.
    def _format_data_response(self, data):
        ndim = data.ndim
        shape = data.shape
        big_endian = data.byteswap()
        enc_data = base64.b64encode(big_endian.tobytes())
        return {'ndim': ndim, 'shape': shape, 'data': enc_data, 'dtype': data.dtype.name}

    def _format_unspecified_error(self, e):
        traceback.print_exc()
        err_dict = {'args': e.args, 'class': e.__class__.__name__, 'doc': e.__doc__, 'message': e.message, 'traceback': traceback.format_exc()}
        raise error.UnspecifiedError(json.dumps(err_dict, encoding='utf-8'))


if __name__ == '__main__':
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

    datacube = dict()
    database = dict()

    # load cell types data, database and dispatch
    data = None
    if not os.path.exists(DATA_DIR + 'cell_types.npy'):
        print('Loading cell_types datacube from warehouse ...')

        cursor.execute("select array_length(col_ids, 1) from data_cube_runs where name = 'cell_types'")
        num_cols = cursor.fetchall()[0][0]

        cursor.execute("""select count(*) from data_cubes dc
            join data_cube_runs dcr on dc.data_cube_run_id = dcr.id
            where dcr.name = 'cell_types'""")
        num_rows = cursor.fetchall()[0][0]

        data = np.full((num_rows, num_cols), np.nan, dtype=np.float32)

        cursor.execute("select data from data_cubes dc join data_cube_runs dcr on dc.data_cube_run_id = dcr.id where dcr.name = '%s'" % 'cell_types')

        progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', Counter(), '/' + str(num_rows) + ' ', ETA(), ' ', FileTransferSpeed(unit='rows')], maxval=num_rows)
        progress.start()

        BATCH_SIZE=50
        row_idx = 0
        while True:
            chunk = cursor.fetchmany(BATCH_SIZE)
            if 0 == len(chunk):
                break
            else:
                data[row_idx:row_idx+len(chunk),:] = np.asarray(chunk, dtype=np.float32)[:,0]
                row_idx += len(chunk)
                progress.update(row_idx)
        progress.finish()

        np.save(DATA_DIR + 'cell_types.npy', data)
    else:
        print('Loading cell_types datacube from filesystem ...')
        data = np.load(DATA_DIR + 'cell_types.npy').astype(np.float32)

    datacube['cell_types'] = Datacube(data, distributed=args.distributed, observed=~np.isnan(data))
    database['cell_types'] = Database('postgresql+pg8000://' + DB_USER + ':' + DB_PASSWORD + '@' + DB_HOST + ':' + str(DB_PORT) + '/' + DB_NAME)

    # load cam data, database and dispatch
    data = None
    if not os.path.exists(DATA_DIR + 'cam.npy'):
        print('Loading cam datacube from lims ...')
        data = cam.load()
        np.save(DATA_DIR + 'cam.npy', data)
    else:
        print('Loading cam datacube from filesystem ...')
        data = np.load(DATA_DIR + 'cam.npy').astype(np.float32)

    datacube['cam'] = Datacube(data, distributed=args.distributed, observed=~np.isnan(data))
    database['cam'] = Database('postgresql+pg8000://' + DB_USER + ':' + DB_PASSWORD + '@' + DB_HOST + ':' + str(DB_PORT) + '/' + DB_NAME)

    # logging
    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    # wamp
    runner = ApplicationRunner(CROSSBAR_ROUTER_URL, WAMP_REALM_NAME)
    runner.run(DatacubeComponent)
