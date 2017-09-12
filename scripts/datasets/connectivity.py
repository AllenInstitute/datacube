#!/usr/bin/env python2

from __future__ import division
import argparse
import os
import logging
import json

import nrrd
import numpy as np
import xarray as xr

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.config.manifest import Manifest

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree


def main():


    manifest_dir = os.path.join(args.data_dir, 'mouse_connectivity_cache')
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)
    manifest_path = os.path.join(manifest_dir, args.manifest_filename)
    mcc = MouseConnectivityCache(manifest_file=manifest_path, resolution=args.resolution, base_uri=args.data_src)

    #todo: this gets secondary structures using source search (backed by existing conn-service). don't do this.
    experiments = mcc.get_experiments()
    experiment_ids = [exp['id'] for exp in experiments]

    if args.fraction_of_experiments != 1:
        step = int(np.around(1.0 / args.fraction_of_experiments))
        experiment_ids = experiment_ids[::step]
        logging.info('using {0} of {1} experiments'.format(len(experiment_ids), len(experiments)))

    for ii, eid in enumerate(experiment_ids):

        data, header = mcc.get_projection_density(eid)
        logging.info('read in data for experiment {0}'.format(eid))

        if ii == 0:
            dshape = list(data.shape)
            volume = np.zeros(dshape + [len(experiment_ids)], dtype=np.float32)
            logging.info('volume occupies {0} bytes ({1})'.format(volume.nbytes, volume.dtype.name))

        volume[:, :, :, ii] = data
        logging.info('pasted data from experiment {0} ({1})'.format(eid, ii)) 


    oapi = OntologiesApi(base_uri=args.data_src)
    structure_graph = oapi.get_structures_with_sets([1])
    structure_graph = StructureTree.clean_structures(structure_graph)
    tree = StructureTree(structure_graph)

    ccf_anno = nrrd.read('/projects/0378/vol1/informatics/model/P56/atlases/MouseCCF2016/annotation_100.nrrd')[0]

    structure_paths = {s['id']: s['structure_id_path'] for s in tree.get_structures_by_id(np.unique(ccf_anno))}
    ontology_depth = max([len(p) for p in structure_paths.values()])
    ccf_anno_paths = np.zeros(ccf_anno.shape+(ontology_depth,), dtype=ccf_anno.dtype)

    for i in range(ccf_anno.shape[0]):
        for j in range(ccf_anno.shape[1]):
            for k in range(ccf_anno.shape[2]):
                structure_id = ccf_anno[i,j,k]
                if structure_id > 0:
                    path = structure_paths[structure_id]
                    ccf_anno_paths[i,j,k,:len(path)] = path


    #todo: integrate scripts used to generate these nc files
    pv = xr.open_dataset('/data/aibstemp/chrisba/pv_tables.nc')
    npv = xr.open_dataset('/data/aibstemp/chrisba/npv_tables.nc')
    projection_unionize = xr.concat([pv,npv],xr.DataArray([False,True],dims=['normalized'],name='normalized'))
    projection_unionize['injection'] = xr.DataArray([False,True], dims=['injection'],name='injection')
    projection_unionize = projection_unionize.rename({'__xarray_dataarray_variable__': 'density'})


    ccf_dims = ['anterior_posterior', 'superior_inferior', 'left_right']
    ds = xr.Dataset(
        data_vars={
            'ccf_structure': (ccf_dims, ccf_anno),
            'ccf_structures': (ccf_dims+['depth'], ccf_anno_paths),
            'projection': (ccf_dims+['experiment'], volume),
            'mouse_line': (['experiment'], [e['transgenic-line'] for e in experiments]),
            'tracer': (['experiment'], [e['product-id'] for e in experiments]) #todo: this is probably wrong
        },
        coords={
            'experiment': experiment_ids
        }
    )
    ds = xr.merge([ds, projection_unionize], join='exact')
    ds.to_netcdf(os.path.join(args.data_dir, args.data_name + '.nc'), format='NETCDF4')


if __name__ == '__main__':

    logging.getLogger('').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Connectivity datacube generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mouse_ccf', help="base name with which to create files")
    parser.add_argument('--manifest_filename', type=str, default='mouse_connectivity_manifest.json')
    parser.add_argument('--resolution', type=int, default=100)
    parser.add_argument('--fraction_of_experiments', type=float, default=1.0)

    args = parser.parse_args()

    main()
