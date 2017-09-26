#!/usr/bin/env python

from __future__ import division
import argparse
import os
import logging
import json
import urllib
import requests
from io import StringIO

import nrrd
import numpy as np
import xarray as xr
import pandas as pd

from util import pd_dataframe_to_np_structured_array, np_structured_array_to_xr_dataset
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.config.manifest import Manifest


def main():

    mcc_dir = os.path.join(args.data_dir, 'mouse_connectivity_cache')
    if not os.path.exists(mcc_dir):
        os.makedirs(mcc_dir)
    manifest_path = os.path.join(mcc_dir, args.manifest_filename)
    all_unionizes_path = os.path.join(mcc_dir, 'all_unionizes.csv')
    mcc = MouseConnectivityCache(manifest_file=manifest_path, resolution=args.resolution, base_uri=args.data_src)

    r=requests.get(args.data_src + '/api/v2/data/ApiConnectivity/query.csv?num_rows=all')
    experiments = pd.read_csv(StringIO(r.text), true_values=['t'], false_values=['f'])

    experiment_ids = list(experiments['data_set_id'])
    tree = mcc.get_structure_tree()
    structure_ids = list(tree.node_ids())

    if not args.annotation_volume_dir:
        ccf_anno = mcc.get_annotation_volume(file_name=os.path.join(mcc_dir, 'annotation_100.nrrd'))[0]
    else:
        ccf_anno = nrrd.read(os.path.join(args.annotation_volume_dir, 'annotation_100.nrrd'))[0]
    structure_paths = {s['id']: s['structure_id_path'] for s in tree.filter_nodes(lambda x: True)}
    ontology_depth = max([len(p) for p in structure_paths.values()])

    def make_annotation_volume_paths():
        ccf_anno_paths = np.zeros(ccf_anno.shape+(ontology_depth,), dtype=ccf_anno.dtype)
        for i in range(ccf_anno.shape[0]):
            for j in range(ccf_anno.shape[1]):
                for k in range(ccf_anno.shape[2]):
                    structure_id = ccf_anno[i,j,k]
                    if structure_id > 0:
                        path = structure_paths[structure_id]
                        ccf_anno_paths[i,j,k,:len(path)] = path
        return ccf_anno_paths
    ccf_anno_paths = make_annotation_volume_paths()

    def make_unionize_tables(data_field_key):
        HEMISPHERE_IDS = [1, 2, 3]
        HEMISPHERE_MAP = {1: 'left', 2: 'right', 3: 'bilateral'}

        def get_projection_table(unionizes, experiment_ids, structure_ids, data_field):
            unionizes = unionizes.pivot(index='experiment_id', columns='structure_id', values=data_field)
            unionizes = unionizes.reindex(index=experiment_ids, columns=structure_ids, fill_value=0.0)
            unionizes = unionizes.fillna(0.0)
            return unionizes

        def get_specified_projection_table(mcc, unionizes, experiment_ids, structure_ids,
                                           is_injection, hemisphere_id, data_field):
            unionizes = mcc.filter_structure_unionizes(unionizes, is_injection=bool(is_injection),
                                                       structure_ids=structure_ids, include_descendants=True,
                                                       hemisphere_ids=[hemisphere_id])
            return get_projection_table(unionizes, experiment_ids, structure_ids, data_field)

        def get_all_unionizes(mcc, all_unionizes_path, experiment_ids):
            try:
                unionizes = pd.read_csv(all_unionizes_path)
            except (IOError, ValueError) as err:
                unionizes = mcc.get_structure_unionizes(experiment_ids)
                unionizes.to_csv(all_unionizes_path)
            return unionizes

        def map_hemisphere_id(hem_id):
           return HEMISPHERE_MAP[hem_id]

        unionizes = get_all_unionizes(mcc, all_unionizes_path, experiment_ids)

        nstructures = len(structure_ids)
        nhemispheres = len(HEMISPHERE_IDS)
        nexperiments = len(experiment_ids)

        tables = xr.DataArray(np.zeros([nexperiments, nstructures, nhemispheres, 2]),
                              dims=['experiment', 'structure', 'hemisphere', 'injection'],
                              coords={'experiment': experiment_ids, 'structure': structure_ids,
                                      'hemisphere': list(map(map_hemisphere_id, HEMISPHERE_IDS)), 'injection': [0, 1]})
        print(tables.coords)

        for hid in HEMISPHERE_IDS:
            for isinj in [0, 1]:
                spec_table = get_specified_projection_table(mcc, unionizes, experiment_ids, structure_ids, isinj, hid, data_field_key)
                print(hid, isinj, spec_table.values.sum())
                tables.loc[:, :, map_hemisphere_id(hid), isinj] =  spec_table

        print(tables.loc[482578964, 1066, 'left', 0])
        print(tables.shape)
        return tables
    pv = make_unionize_tables('projection_volume')
    npv = make_unionize_tables('normalized_projection_volume')

    projection_unionize = xr.concat([pv,npv],xr.DataArray([False,True],dims=['normalized'],name='normalized'))
    projection_unionize['injection'] = xr.DataArray([False,True], dims=['injection'],name='injection')

    def make_structure_paths_array():
        structure_paths_array = np.zeros(projection_unionize.structure.shape+(ontology_depth,), dtype=projection_unionize.structure.dtype)
        for i in range(projection_unionize.structure.shape[0]):
            structure_id = int(projection_unionize.structure[i])
            if structure_id > 0:
                path = structure_paths[structure_id]
                structure_paths_array[i,:len(path)] = path
        return structure_paths_array
    structure_paths_array = make_structure_paths_array()

    def make_projection_volume():
        for ii, eid in enumerate(experiment_ids):

            data, header = mcc.get_projection_density(eid)
            logging.info('read in data for experiment {0}'.format(eid))

            if ii == 0:
                dshape = list(data.shape)
                volume = np.zeros(dshape + [len(experiment_ids)], dtype=np.float32)
                logging.info('volume occupies {0} bytes ({1})'.format(volume.nbytes, volume.dtype.name))

            volume[:, :, :, ii] = data
            logging.info('pasted data from experiment {0} ({1})'.format(eid, ii))
        return volume
    volume = make_projection_volume()

    ccf_dims = ['anterior_posterior', 'superior_inferior', 'left_right']
    ds = xr.Dataset(
        data_vars={
            'ccf_structure': (ccf_dims, ccf_anno),
            'ccf_structures': (ccf_dims+['depth'], ccf_anno_paths),
            'projection': (ccf_dims+['experiment'], volume),
            'volume': projection_unionize
        },
        coords={
            'experiment': experiment_ids,
            'structures': (['structure', 'depth'], structure_paths_array),
            'anterior_posterior': 100*np.arange(ccf_anno.shape[0]),
            'superior_inferior': 100*np.arange(ccf_anno.shape[1]),
            'left_right': 100*np.arange(ccf_anno.shape[2])
        }
    )
    experiments_sa = pd_dataframe_to_np_structured_array(experiments)
    experiments_ds = np_structured_array_to_xr_dataset(experiments_sa)
    experiments_ds.rename({'dim_0': 'experiment'}, inplace=True)
    ds.merge(experiments_ds, inplace=True)
    ds.to_netcdf(os.path.join(args.data_dir, args.data_name + '.nc'), format='NETCDF4')


if __name__ == '__main__':

    logging.getLogger('').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Connectivity datacube generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mouse_ccf', help="base name with which to create files")
    parser.add_argument('--annotation-volume-dir', default=None, help="directory from which to get the CCF annotation NRRD; get from the sdk if omitted")
    parser.add_argument('--manifest_filename', type=str, default='mouse_connectivity_manifest.json')
    parser.add_argument('--resolution', type=int, default=100)

    args = parser.parse_args()

    main()
