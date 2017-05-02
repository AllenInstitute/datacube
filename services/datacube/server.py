import txaio
from pprint import pprint
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.types import RegisterOptions
import numpy as np
import json
import base64
import traceback
import argparse
import pg8000
import os
import sys
import StringIO
from PIL import Image

from datacube import Datacube
#import cam
import error


DATA_DIR = './data/'

DB_HOST = 'testwarehouse1'
DB_PORT = 5432
DB_NAME = 'warehouse-R193'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'

DEMO_ROUTER_URL = u"ws://127.0.0.1:8082/ws"
WAMP_REALM_NAME = u"aibs"

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
                self._format_error(e)

        def raw(select, cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].raw(select)
                return self._format_data_response(data)
            except Exception as e:
                self._format_error(e)

        def log2_1p(select, cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].log2_1p(select)
                return self._format_data_response(data)
            except Exception as e:
                self._format_error(e)

        def standard(select, axis, cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].standard(select, axis)
                return self._format_data_response(data)
            except Exception as e:
                self._format_error(e)

        def image(select, image_format='jpeg', cube=None):
            try:
                cube = self._select_cube(cube)
                data = datacube[cube].raw(select)
                data = np.squeeze(data)

                if data.ndim != 2 and not (data.ndim == 3 and data.shape[2] == 4):
                    raise error.Non2dImageError()

                if data.ndim == 2 and data.dtype != np.uint8:
                    data = ((data / datacube[cube].max) * 255.0).astype(np.uint8)

                image = Image.fromarray(data)
                buf = StringIO.StringIO()
                if image_format.lower() == 'jpeg':
                    image.save(buf, format='JPEG', quality=40)
                elif image_format.lower() == 'png':
                    image.save(buf, format='PNG')
                else:
                    raise error.InvalidOrUnsupportedImageFormat()
                #return {'data': bytes(buf.getvalue())}
                return {'data': 'data:image/' + image_format.lower() + ';base64,' + base64.b64encode(buf.getvalue())}
            except Exception as e:
                self._format_error(e)

        def corr(select, axis, seed, cube=None):
            try:
                cube = self._select_cube(cube)
                return datacube[cube].corr(select, axis, seed)
            except Exception as e:
                self._format_error(e)

        def fold_change(numerator, denominator, axis, cube=None):
            try:
                cube = self._select_cube(cube)
                return datacube[cube].fold_change(numerator, denominator, axis)
            except Exception as e:
                self._format_error(e)

        def diff(numerator, denominator, axis, cube=None):
            try:
                cube = self._select_cube(cube)
                return datacube[cube].diff(numerator, denominator, axis)
            except Exception as e:
                self._format_error(e)

        def meta(cube=None):
            try:
                cube_name = cube
                cube = self._select_cube(cube)

                col_meta_file = DATA_DIR+cube_name+'_cols.json.zz.b64'
                row_meta_file = DATA_DIR+cube_name+'_rows.json.zz.b64'
                meta_dict = {}
                if os.path.exists(col_meta_file):
                    meta_dict['cols'] = open(col_meta_file).read()
                if os.path.exists(row_meta_file):
                    meta_dict['rows'] = open(row_meta_file).read()
                return meta_dict
            except Exception as e:
                self._format_error(e)

        try:
            yield self.register(raw, u'org.alleninstitute.datacube.raw')
            yield self.register(log2_1p, u'org.alleninstitute.datacube.log2_1p')
            yield self.register(standard, u'org.alleninstitute.datacube.standard')
            yield self.register(image, u'org.alleninstitute.datacube.image')
            yield self.register(info, u'org.alleninstitute.datacube.info')
            yield self.register(corr, u'org.alleninstitute.datacube.corr')
            yield self.register(fold_change, u'org.alleninstitute.datacube.fold_change')
            yield self.register(diff, u'org.alleninstitute.datacube.diff')
            yield self.register(meta, u'org.alleninstitute.datacube.meta')
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
        #enc_data = base64.b64encode(big_endian.tobytes())
        return {'ndim': ndim, 'shape': shape, 'data': bytes(big_endian.tobytes()), 'dtype': unicode(data.dtype.name)}


    def _format_error(self, e):
        traceback.print_exc()
        err_dict = {'error': True, 'class': unicode(e.__class__.__name__), 'doc': unicode(e.__doc__), 'message': unicode(e.message), 'traceback': unicode(traceback.format_exc())}
        raise error.UnspecifiedError(err_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--demo', action='store_true', dest='demo', help='load demo dataset')
    parser.set_defaults(demo=False)
    #parser.add_argument('--include-cubes', type=str)
    args = parser.parse_args()
    #args.include_cubes = args.include_cubes.split(',')

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    datacube = dict()

    if args.demo:
        #from progressbar import ProgressBar, Percentage, Bar, ETA, Counter, FileTransferSpeed

        ## load cell types data
        #conn = pg8000.connect(user=DB_USER, host=DB_HOST, port=DB_PORT, database=DB_NAME, password=DB_PASSWORD)
        #cursor = conn.cursor()
        #cursor.execute("select name from data_cube_runs")
        #results = cursor.fetchall()
        ##available_cubes = str(results[0])

        #data = None
        #if not os.path.exists(DATA_DIR + 'cell_types.npy'):
        #    print('Loading cell_types datacube from warehouse ...')

        #    cursor.execute("select array_length(col_ids, 1) from data_cube_runs where name = 'cell_types'")
        #    num_cols = cursor.fetchall()[0][0]

        #    cursor.execute("""select count(*) from data_cubes dc
        #        join data_cube_runs dcr on dc.data_cube_run_id = dcr.id
        #        where dcr.name = 'cell_types'""")
        #    num_rows = cursor.fetchall()[0][0]

        #    data = np.full((num_rows, num_cols), np.nan, dtype=np.float32)

        #    cursor.execute("select data from data_cubes dc join data_cube_runs dcr on dc.data_cube_run_id = dcr.id where dcr.name = '%s'" % 'cell_types')

        #    progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', Counter(), '/' + str(num_rows) + ' rows ', ETA(), ' ', FileTransferSpeed(unit='rows')], maxval=num_rows)
        #    progress.start()

        #    BATCH_SIZE=50
        #    row_idx = 0
        #    while True:
        #        chunk = cursor.fetchmany(BATCH_SIZE)
        #        if 0 == len(chunk):
        #            break
        #        else:
        #            data[row_idx:row_idx+len(chunk),:] = np.asarray(chunk, dtype=np.float32)[:,0]
        #            row_idx += len(chunk)
        #            progress.update(row_idx)
        #    progress.finish()

        #    np.save(DATA_DIR + 'cell_types.npy', data)
        #else:
        #    print('Loading cached cell_types datacube from ./data/cell_types.npy ...')
        #    data = np.load(DATA_DIR + 'cell_types.npy').astype(np.float32)

        #datacube['cell_types'] = Datacube(data, distributed=args.distributed, observed=~np.isnan(data))

        ## load cam data
        #data = None
        #if not os.path.exists(DATA_DIR + 'cam.npy'):
        #    print('Loading cam datacube from lims ...')
        #    data = cam.load()
        #    np.save(DATA_DIR + 'cam.npy', data)
        #else:
        #    print('Loading cached cam datacube from ./data/cam.npy ...')
        #    data = np.load(DATA_DIR + 'cam.npy').astype(np.float32)

        #datacube['cam'] = Datacube(data, distributed=args.distributed, observed=~np.isnan(data))

        # MNI demo

        import nrrd
        import urllib2
        from numba import jit

        if not os.path.exists(DATA_DIR + 'ccf.npy'):
            print('Loading CCF atlas volume from informatics directory')
            ccf = nrrd.read('/projects/0378/vol1/informatics/model/P56/atlasVolume/average_template_25.nrrd')[0]
            ccf = (ccf.astype(np.float32)/516.0*255.0).astype(np.uint8)
            np.save(DATA_DIR + 'ccf.npy', ccf)
            del ccf

        if not os.path.exists(DATA_DIR + 'ccf_anno.npy') or not os.path.exists(DATA_DIR + 'ccf_anno_color.npy'):
            print('Loading CCF annotation volume from informatics directory')
            ccf_anno = nrrd.read('/projects/0378/vol1/informatics/model/P56/atlases/MouseCCF2016/annotation_25.nrrd')[0]
            np.save(DATA_DIR + 'ccf_anno.npy', ccf_anno)

            print('Generating colorized annotation volume')
            def get_structure_colors(ccf_anno):
                annotated_structures = np.unique(ccf_anno)
                structure_colors = dict()
                structure_colors[0] = np.array([0, 0, 0, 0], dtype=np.uint8)
                for structure_id in annotated_structures:
                    if structure_id:
                        STRUCTURE_API = 'http' + '://' + 'api.brain-map.org/api/v2/data/Structure/query.json?id=' # todo
                        color_hex_triplet = json.load(urllib2.urlopen(STRUCTURE_API + str(structure_id)))['msg'][0]['color_hex_triplet']
                        structure_colors[int(structure_id)] = np.concatenate([np.array(bytearray(color_hex_triplet.decode("hex"))), [255]]).astype(np.uint8)
                return structure_colors
                        
            structure_colors = get_structure_colors(ccf_anno)

            ccf_anno_color = np.zeros(ccf_anno.shape+(4,), dtype=np.uint8)

            @jit(nopython=True)
            def colorize(ccf_anno, ids, colors, ccf_anno_color):
                for i in range(ccf_anno.shape[0]):
                    for j in range(ccf_anno.shape[1]):
                        for k in range(ccf_anno.shape[2]):
                            color_idx = np.searchsorted(ids, ccf_anno[i,j,k])
                            ccf_anno_color[i,j,k,:] = colors[color_idx]

            sorted_colors = sorted(structure_colors.items(), key=lambda x: x[0])
            ids = map(lambda x: x[0], sorted_colors)
            colors = np.stack(map(lambda x: x[1], sorted_colors), axis=0)
            colorize(ccf_anno, ids, colors, ccf_anno_color)
            np.save(DATA_DIR + 'ccf_anno_color.npy', ccf_anno_color)

            del ccf_anno
            del ccf_anno_color

        ccf = np.load(DATA_DIR + 'ccf.npy', mmap_mode='r')
        datacube['ccf'] = Datacube(ccf, distributed=args.distributed)
        ccf_anno = np.load(DATA_DIR + 'ccf_anno.npy', mmap_mode='r')
        datacube['ccf_anno'] = Datacube(ccf_anno, distributed=args.distributed)
        ccf_anno_color = np.load(DATA_DIR + 'ccf_anno_color.npy', mmap_mode='r')
        datacube['ccf_anno_color'] = Datacube(ccf_anno_color, distributed=args.distributed)
    else:
        print('TODO: implement non-demo mode (load files from data-dir specified on command line)')
        exit(0)

    # logging
    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    # wamp
    runner = ApplicationRunner(DEMO_ROUTER_URL, WAMP_REALM_NAME)
    runner.run(DatacubeComponent)
