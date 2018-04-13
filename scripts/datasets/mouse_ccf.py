#!/usr/bin/env python

import argparse
import nrrd
import numpy as np
import xarray as xr
import os.path
import urllib.request
from numba import jit
import json
from future.utils import lmap

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.config.manifest import Manifest


def main():
    parser = argparse.ArgumentParser(description='MNI reference space datacube generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mouse_ccf', help="base name with which to create files")
    parser.add_argument('--manifest_filename', type=str, default='mouse_connectivity_manifest.json')
    parser.add_argument('--resolution', type=int, default=25)
    args = parser.parse_args()

    generate(args.data_src, args.data_dir, args.data_name, args.manifest_filename, args.resolution)


def generate(data_src, data_dir, data_name, manifest_filename, resolution):
    print('Generating...')
    data_path = os.path.join(data_dir, data_name + '.nc')
    if not os.path.exists(data_path):
        mcc_dir = os.path.join(data_dir, 'mouse_connectivity_cache')
        if not os.path.exists(mcc_dir):
            os.makedirs(mcc_dir)
        manifest_path = os.path.join(mcc_dir, manifest_filename)
        mcc = MouseConnectivityCache(manifest_file=manifest_path, resolution=resolution, base_uri=data_src)

        ccf = mcc.get_template_volume(file_name=os.path.join(mcc_dir, 'average_template_{:d}.nrrd'.format(resolution)))[0]
        ccf = (ccf.astype(np.float32)/np.max(ccf)*255.0).astype(np.uint8)
        ccf_anno = mcc.get_annotation_volume(file_name=os.path.join(mcc_dir, 'annotation_{:d}.nrrd'.format(resolution)))[0]

        def get_structure_colors(ccf_anno):
            annotated_structures = np.unique(ccf_anno)
            structure_colors = dict()
            structure_colors[0] = np.array([0, 0, 0, 0], dtype=np.uint8)
            #todo: maybe could use allensdk OntologiesApi and StructureTree.get_colormap() instead of this
            for structure_id in annotated_structures:
                if structure_id:
                    STRUCTURE_API = data_src + '/api/v2/data/Structure/query.json?id='
                    color_hex_triplet = json.load(urllib.request.urlopen(STRUCTURE_API + str(structure_id)))['msg'][0]['color_hex_triplet']
                    structure_colors[int(structure_id)] = np.concatenate([np.array(bytearray.fromhex(color_hex_triplet)), [255]]).astype(np.uint8)
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
        ids = lmap(lambda x: x[0], sorted_colors)
        colors = np.stack(lmap(lambda x: x[1], sorted_colors), axis=0)
        colorize(ccf_anno, ids, colors, ccf_anno_color)

        dims = ['anterior_posterior', 'superior_inferior', 'left_right']
        ds = xr.Dataset(
            data_vars={
                'ccf': (dims, ccf, {'spacing': [resolution]*3}),
                'annotation': (dims, ccf_anno),
                'color': (dims+['RGBA'], ccf_anno_color)
            },
            coords={
                'anterior_posterior': resolution*np.arange(ccf.shape[0]),
                'superior_inferior': resolution*np.arange(ccf.shape[1]),
                'left_right': resolution*np.arange(ccf.shape[2])
            }
        )
        ds.to_netcdf(data_path, format='NETCDF4')
    print('Data created in data_dir.')


if __name__ == '__main__':
    main()

