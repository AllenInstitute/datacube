#!/usr/bin/env python

from __future__ import division
import argparse
import os
import logging
import json
import urllib
import requests
from io import StringIO
import shutil

import nrrd
import numpy as np
import xarray as xr
import pandas as pd

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.config.manifest import Manifest


FIBER_TRACTS_ID = 1009
SUMMARY_SET_ID = 167587189
HEMISPHERE_IDS = [1, 2, 3]
HEMISPHERE_MAP = {1: 'left', 2: 'right', 3: 'bilateral'}
API_CONNECTIVITY_QUERY = '/api/v2/data/ApiConnectivity/query.csv?num_rows=all'


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


def make_unionize_tables(data_field_key):

    unionizes = get_all_unionizes(mcc, all_unionizes_path, experiment_ids)

    nstructures = len(structure_ids)
    nhemispheres = len(HEMISPHERE_IDS)
    nexperiments = len(experiment_ids)

    tables = xr.DataArray(np.zeros([nexperiments, nstructures, nhemispheres, 2]),
                            dims=['experiment', 'structure', 'hemisphere', 'injection'],
                            coords={'experiment': experiment_ids, 'structure': structure_ids,
                                    'hemisphere': list(map(map_hemisphere_id, HEMISPHERE_IDS)), 'injection': [False, True]})
    print(tables.coords)

    for hid in HEMISPHERE_IDS:
        for isinj in [0, 1]:
            spec_table = get_specified_projection_table(mcc, unionizes, experiment_ids, structure_ids, isinj, hid, data_field_key)
            print(hid, isinj, spec_table.values.sum())
            tables.loc[:, :, map_hemisphere_id(hid), isinj] =  spec_table

    print(tables.loc[482578964, 1066, 'left', 0])
    print(tables.shape)
    return tables


def make_structure_paths_array():
    structure_paths_array = np.zeros(projection_unionize.structure.shape+(ontology_depth,), dtype=projection_unionize.structure.dtype)
    for i in range(projection_unionize.structure.shape[0]):
        structure_id = int(projection_unionize.structure[i])
        if structure_id > 0:
            path = structure_paths[structure_id]
            structure_paths_array[i,:len(path)] = path
    return structure_paths_array


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


def make_primary_structure_paths(primary_structures, ontology_depth, structure_paths):
    ''' Builds a ragged array of structure paths starting at the primary structure for each experiment.
    '''

    primary_structure_paths = np.zeros((len(primary_structures), ontology_depth), dtype=primary_structures.dtype)

    for i in range(len(primary_structures)):
        structure_id = int(primary_structures[i])
        
        if structure_id > 0:
            path = structure_paths[structure_id]
            primary_structure_paths[i,:len(path)] = path

    return primary_structure_paths


def make_projection_volume(experiment_ids, mcc):
    ''' Build a 4D array of projection density volumes, with experiment as the 4th axis
    '''

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


def make_injection_structures_arrays(experiments_ds, ontology_depth, structure_paths):
    '''
    '''

    injection_structures_list = [
        [int(id) for id in s.split('/')] 
        for s in experiments_ds.injection_structures.values.tolist()
    ]

    injection_structures_arr = np.zeros((len(injection_structures_list), max([len(x) for x in injection_structures_list])))
    injection_structure_paths = np.zeros(injection_structures_arr.shape+(ontology_depth,), dtype=injection_structures_arr.dtype)

    for i, structures in enumerate(injection_structures_list):
        injection_structures_arr[i][:len(structures)] = structures

        for j in range(len(structures)):
            path = structure_paths[structures[j]]
            injection_structure_paths[i, j, :len(path)] = path

    return injection_structures_arr, injection_structure_paths


def rgb_to_hex(rgb):
    ''' Convert an rgb triplet to a hex string
    '''

    return ''.join([ hex(ii)[2:] for ii in rgb ])


def get_structure_information(tree, summary_set_id=SUMMARY_SET_ID, exclude_from_summary={FIBER_TRACTS_ID,}):
    ''' Convenience function for extracting relevant metadata about ontological structures

    Parameters
    ----------
    tree : allensdk.core.structure_tree.StructureTree
        Holds the structure ontology.
    summary_set_id : int, optional
        Use this id to determine which structures are in the summary structure set. Defaults to 167587189.
    exlude_from_summary : set of int, optional
        Exclude these structures from the summary structure set. Defaults to 1009 (fiber tracts).

    Returns
    -------
    structure_meta : xarray.Dataset
        Contains the name and acronym of each structure
    structure_ids : list of int
        The id of each structure
    structure_colors : list of str
        The ontological color of each structure (as a hex string)
    structure_paths : dict | int -> list of int
        Maps structure ids to lists of int. Each such list describes the path from the root of the structure tree to 
        structure in question.
    summary_structures : set of int
        Contains the ids of each summary structure (minus those specifically excluded)

    '''

    nodes = pd.DataFrame(tree.nodes())

    structure_ids = list(nodes['id'].values)
    structure_colors = list(nodes['rgb_triplet'].map(rgb_to_hex).values)
    structure_paths = { row.id: row.structure_id_path for row in nodes.itertuples() }

    summary_structures = {
        s['id'] for s in tree.get_structures_by_set_id([summary_set_id]) 
        if s['id'] not in exclude_from_summary
    }

    structure_meta = nodes[['name','acronym']]
    structure_meta = xr.Dataset.from_dataframe(structure_meta)
    structure_meta = structure_meta.drop('index').rename({'index': 'structure'})

    return structure_meta, structure_ids, structure_colors, structure_paths, summary_structures


def main():

    mcc_dir = os.path.join(args.data_dir, 'mouse_connectivity_cache')
    if not os.path.exists(mcc_dir):
        os.makedirs(mcc_dir)
    manifest_path = os.path.join(mcc_dir, args.manifest_filename)
    all_unionizes_path = os.path.join(mcc_dir, 'all_unionizes.csv')
    mcc = MouseConnectivityCache(manifest_file=manifest_path, resolution=args.resolution, base_uri=args.data_src)

    tree = mcc.get_structure_tree()
    structure_meta, structure_ids, structure_colors, structure_paths, summary_structures = get_structure_information(tree)

    r=requests.get(args.data_src + API_CONNECTIVITY_QUERY)
    experiments = pd.read_csv(StringIO(r.text), true_values=['t'], false_values=['f'])

    experiments_ds = xr.Dataset.from_dataframe(experiments)
    experiments_ds.rename({'index': 'experiment'}, inplace=True)
    experiments_ds.coords['experiment'] = experiments_ds.data_set_id

    experiment_ids = experiments_ds.data_set_id.values.tolist()

    ccf_anno = mcc.get_annotation_volume(file_name=os.path.join(mcc_dir, 'annotation_{:d}.nrrd'.format(args.resolution)))[0]
    ontology_depth = max([len(p) for p in structure_paths.values()])
    ccf_anno_paths = make_annotation_volume_paths()

    pv = make_unionize_tables('projection_volume')
    npv = make_unionize_tables('normalized_projection_volume')

    projection_unionize = xr.concat([pv,npv],xr.DataArray([False,True],dims=['normalized'],name='normalized'))
    structure_volumes = make_unionize_tables('volume')

    structure_paths_array = make_structure_paths_array()

    primary_structures = experiments_ds.structure_id.values
    primary_structure_paths = make_primary_structure_paths(primary_structures, ontology_depth, structure_paths)

    volume = make_projection_volume(experiment_ids, mcc)

    injection_structures_arr, injection_structure_paths = make_injection_structures_arrays(
        experiments_ds, ontology_depth, structure_paths)

    ccf_dims = ['anterior_posterior', 'superior_inferior', 'left_right']
    ds = xr.Dataset(
        data_vars={
            'ccf_structure': (ccf_dims, ccf_anno, {'spacing': [args.resolution]*3}),
            'ccf_structures': (ccf_dims+['depth'], ccf_anno_paths),
            'projection': (ccf_dims+['experiment'], volume),
            'volume': projection_unionize,
            'structure_volumes': structure_volumes,
            'primary_structures': (['experiment', 'depth'], primary_structure_paths),
            'injection_structures_array': (['experiment', 'secondary'], injection_structures_arr),
            'injection_structure_paths': (['experiment', 'secondary', 'depth'], injection_structure_paths)
        },
        coords={
            'experiment': experiment_ids,
            'structures': (['structure', 'depth'], structure_paths_array),
            'is_summary_structure': (['structure'], [structure in summary_structures for structure in projection_unionize.structure]),
            'structure_color': structure_colors,
            'anterior_posterior': args.resolution*np.arange(ccf_anno.shape[0]),
            'superior_inferior': args.resolution*np.arange(ccf_anno.shape[1]),
            'left_right': args.resolution*np.arange(ccf_anno.shape[2]),
        }
    )

    ds.merge(experiments_ds, inplace=True, join='exact')
    ds.merge(structure_meta, inplace=True, join='left')
    ds['is_primary'] = (ds.structure_id==ds.structures).any(dim='depth') #todo: make it possible to do this masking on-the-fly
    ds.to_netcdf(os.path.join(args.data_dir, args.data_name + '.nc'), format='NETCDF4', engine='h5netcdf')

    #PBS-1262:
    surface_coords_file = '/allen/programs/celltypes/production/0378/informatics/model/P56/corticalCoordinates/surface_coords_10.h5'
    shutil.copy2(surface_coords_file, args.data_dir)


if __name__ == '__main__':

    logging.getLogger('').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Connectivity datacube generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mouse_ccf', help="base name with which to create files")
    parser.add_argument('--manifest_filename', type=str, default='mouse_connectivity_manifest.json')
    parser.add_argument('--resolution', type=int, default=100)

    args = parser.parse_args()

    main()
