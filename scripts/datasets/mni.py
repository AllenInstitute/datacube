#!/usr/bin/env python

import argparse
import nrrd
import numpy as np
import xarray as xr
import os.path


def main():
    parser = argparse.ArgumentParser(description='MNI reference space datacube generator script')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mni', help="base name with which to create files")
    args = parser.parse_args()

    generate(args.data_dir, args.data_name)


def generate(data_dir='./', data_name='mni'):
    print('Generating...')
    data_path = os.path.join(data_dir, data_name + '.nc')
    if not os.path.exists(data_path):
        mni,opts = nrrd.read('/data/mat/NileG/informatics/data/june_2017/human_ccf/mni_icbm152_t2_tal_nlin_sym_09b_hires.nrrd')
        mni_anno,_ = nrrd.read('/data/mat/NileG/informatics/data/june_2017/human_ccf/mirrored_human_annotation.nrrd')
        mni_anno_color,_ = nrrd.read('/data/mat/NileG/informatics/data/june_2017/human_ccf/colored_annotation.nrrd')
        mni_anno_color = np.moveaxis(mni_anno_color, 0, 3)
        mni_anno_color = np.concatenate((mni_anno_color, (255*np.any(mni_anno_color>0, axis=3, keepdims=True)).astype(np.uint8)), axis=3)

        dims = ['left_right', 'anterior_posterior', 'superior_inferior']
        ds = xr.Dataset(data_vars={'mni': (dims, mni), 'annotation': (dims, mni_anno), 'color': (dims+['RGBA'], mni_anno_color)},
                        coords={
                            'mni_left_right': (['left_right'], (np.array(range(mni.shape[0]))*float(opts['space directions'][0][0]))+float(opts['space origin'][0])),
                            'mni_anterior_posterior': (['anterior_posterior'], (np.array(range(mni.shape[1]))*float(opts['space directions'][1][1]))+float(opts['space origin'][1])),
                            'mni_superior_inferior': (['superior_inferior'], (np.array(range(mni.shape[2]))*float(opts['space directions'][2][2]))+float(opts['space origin'][2]))
                        })
        ds.to_netcdf(data_path, format='NETCDF4', engine='h5netcdf')
    print('Data created in data_dir.')


if __name__ == '__main__':
    main()

