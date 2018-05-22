#!/usr/bin/env python

from __future__ import division
import argparse
import os
import logging
import json
import urllib.parse
import requests
from io import StringIO
import shutil
from collections import OrderedDict
import tempfile
import psycopg2
import csv
from concurrent.futures import ThreadPoolExecutor
import functools
from functools import partial
import types

import nrrd
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import numcodecs
import h5py
import dask
import dask.array as da

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.config.manifest import Manifest


FIBER_TRACTS_ID = 1009
SUMMARY_SET_ID = 167587189
HEMISPHERE_MAP = OrderedDict({1: 'left', 2: 'right', 3: 'bilateral'})
API_CONNECTIVITY_QUERY = '/api/v2/data/ApiConnectivity/query.csv?num_rows=all'
DEFAULT_SURFACE_COORDS_PATH = (
    '/allen/programs/celltypes/production/0378/informatics/model/P56/corticalCoordinates/surface_coords_10.h5'
)
WAREHOUSE_DATABASES = json.load(open('warehouse_databases.json'))
UNIONIZE_QUERY = '''
    select section_data_set_id, structure_id, hemisphere_id, is_injection,
    projection_volume, normalized_projection_volume, volume
    from projection_structure_unionizes
    where section_data_set_id in ({})
    '''
DATASET_STORAGE_DIR_QUERY = '''select storage_directory from data_sets where id={}'''



def get_projection_table(unionizes, experiment_ids, structure_ids, data_field):
    ''' Builds an experiments X structures table out of a unionize table.

    Parameters
    ----------
    unionizes : pd.DataFrame
        Table of projection structure unionizes. Each row describes a spatial domain on a particular experiment.
    experiment_ids : list of int
        Use data from these experiments (will become the index of the output dataframe).
    structure_ids : list of int
        Use data from these structures (will become the columns of the output dataframe).
    data_field : str
        Use this field as the values of the projection table

    Returns
    -------
    pd.DataFrame :
        Output table.

    '''

    unionizes = unionizes.pivot(index='experiment_id', columns='structure_id', values=data_field)
    unionizes = unionizes.reindex(index=experiment_ids, columns=structure_ids, fill_value=0.0)
    unionizes = unionizes.fillna(0.0)

    return unionizes


def get_specified_projection_table(
    mcc, unionizes, experiment_ids, structure_ids, is_injection, hemisphere_id, data_field
    ):
    ''' Obtain an experiments X structures table of unionize values, slicing by hemisphere and injection status.
    '''

    unionizes = mcc.filter_structure_unionizes(
        unionizes,
        is_injection=bool(is_injection),
        structure_ids=structure_ids,
        include_descendants=True,
        hemisphere_ids=[hemisphere_id]
    )

    return get_projection_table(unionizes, experiment_ids, structure_ids, data_field)


def get_all_unionizes(mcc, all_unionizes_path, experiment_ids, warehouse=None):
    ''' Reads a table containing all projection structure unionizes for a given list of experiments. If necessary,
    this table will be compiled and stored.

    Parameters
    ----------
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache
        Use this cache to obtain unionizes.
    all_unionizes_path : str
        Read the collated table from here, or if this file does not exists, write it to here.
    experiment_ids : list of int
        Get unionizes for these experiments
    warehouse : psycopg2.Connection or None
        Warehouse database connection to use if running with --internal.

    Returns
    -------
    unionizes : pd.DataFrame
        Collated table of unionizes

    '''

    try:
        logging.info('found all-unionizes csv')
        unionizes = pd.read_csv(all_unionizes_path)
        assert( set(experiment_ids) - set(unionizes['experiment_id'].values) == set([]) )

    except (IOError, ValueError, AssertionError) as err:
        if warehouse:
            logging.info('downloading unionizes from warehouse...')
            cursor = warehouse.cursor()
            cursor.execute(UNIONIZE_QUERY.format(','.join(str(eid) for eid in experiment_ids)))
            with open(all_unionizes_path, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['experiment_id', 'structure_id', 'hemisphere_id', 'is_injection',
                    'projection_volume', 'normalized_projection_volume', 'volume'])
                for row in cursor:
                    writer.writerow(row)
            cursor.close()
            unionizes = pd.read_csv(all_unionizes_path)
            logging.info('done')
        else:
            logging.info('downloading unionizes to mouse connectivity cache...')
            unionizes = mcc.get_structure_unionizes(experiment_ids)
            unionizes.to_csv(all_unionizes_path)
            logging.info('done')

    return unionizes


def make_unionize_tables(data_field_key, mcc, all_unionizes_path, experiment_ids, structure_ids, hemisphere_map=HEMISPHERE_MAP, warehouse=None):
    ''' Build a 4D table of unionize values, organized by experiment, structure, hemisphere and injection status

    Parameters
    ----------
    data_field_key : str
        Use this field as the values in the output table
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache
        Used to manage data access
    all_unionizes_path : str
        Path to csv file containing unionizes for the entire dataset
    experiment_ids : list of int
        Ids of experiments to include
    structure_ids : list of int
        Ids of structures to include
    hemisphere_map : collections.OrderedDict
        Maps hemisphere ids to full names
    warehouse : psycopg2.Connection or None
        Warehouse database connection to use if running with --internal.

    Returns
    -------
    tables : xarray.DataArray
        4D unionize table

    '''

    unionizes = get_all_unionizes(mcc, all_unionizes_path, experiment_ids, warehouse)

    nstructures = len(structure_ids)
    nhemispheres = len(hemisphere_map)
    nexperiments = len(experiment_ids)

    tables = xr.DataArray(
        np.zeros([nexperiments, nstructures, nhemispheres, 2]),
        dims=['experiment', 'structure', 'hemisphere', 'injection'],
        coords={'experiment': experiment_ids, 'structure': structure_ids, 'hemisphere': list(hemisphere_map.values()),
        'injection': [False, True]}
    )
    print(tables.coords)

    for hid in hemisphere_map:
        for isinj in [0, 1]:
            spec_table = get_specified_projection_table(mcc, unionizes, experiment_ids, structure_ids, isinj, hid, data_field_key)
            print(hid, isinj, spec_table.values.sum())
            tables.loc[:, :, hemisphere_map[hid], isinj] =  spec_table

    print(tables.loc[482578964, 1066, 'left', 0])
    print(tables.shape)
    return tables


def make_structure_paths_array(projection_unionize, ontology_depth, structure_paths):
    ''' Builds a ragged array of structure id paths
    '''

    structure_paths_array = np.zeros(
        projection_unionize.structure.shape+(ontology_depth,),
        dtype=projection_unionize.structure.dtype
    )

    for i in range(projection_unionize.structure.shape[0]):
        structure_id = int(projection_unionize.structure[i])

        if structure_id > 0:
            path = structure_paths[structure_id]
            structure_paths_array[i, :len(path)] = path

    return structure_paths_array


def make_annotation_volume_paths(ccf_anno, ontology_depth, structure_paths):
    ''' Builds a 4D array, ragged in the last axis, which assigns to each CCF voxel
    the structure id path associated with the structure at that voxel.

    '''

    ccf_anno_paths = np.zeros(ccf_anno.shape+(ontology_depth,), dtype=ccf_anno.dtype)

    for i in range(ccf_anno.shape[0]):
        for j in range(ccf_anno.shape[1]):
            for k in range(ccf_anno.shape[2]):

                structure_id = ccf_anno[i,j,k]

                if structure_id > 0:
                    path = structure_paths[structure_id]
                    ccf_anno_paths[i, j, k, :len(path)] = path

    return ccf_anno_paths


def make_primary_structure_paths(primary_structures, ontology_depth, structure_paths):
    ''' Builds a ragged array which assigns to each experiment the structure id path associated with that
    experiment's primary structure

    Parameters
    ----------
    primary_structures : array-like of int
        Identifies the primary injection structure of each experiment.
    ontology_depth : int
        Maximum node depth within the ontology tree.
    structure_paths : dict | int -> list of int
        Maps structure ids to lists of int. Each such list describes the path from the root of the structure tree to
        structure in question.

    Returns
    -------
    primary_structure_paths ; numpy.ndarray
        Each row contains the structure id path of a single experiment's primary injection structure
        (right-padded with zeros if necessary).

    '''

    paths_shape = (len(primary_structures), ontology_depth)
    primary_structure_paths = np.zeros(paths_shape, dtype=primary_structures.dtype)

    for i in range(len(primary_structures)):
        structure_id = int(primary_structures[i])

        if structure_id > 0:
            path = structure_paths[structure_id]
            primary_structure_paths[i,:len(path)] = path

    return primary_structure_paths


def get_projection_density_grid_data(experiment_id, mcc, warehouse=None):
    ''' Get 3D projection density volume by downloading nrrd through AllenSDK or reading
    from the filesystem (--internal) '''

    if warehouse:
        cursor = warehouse.cursor()
        cursor.execute(DATASET_STORAGE_DIR_QUERY.format(experiment_id))
        storage_directory = cursor.fetchone()[0]
        cursor.close()
        nrrd_path = os.path.join(storage_directory, 'grid', 'projection_density_{}.nrrd'.format(mcc.resolution))
        data, header = nrrd.read(nrrd_path)
    else:
        data, header = mcc.get_projection_density(experiment_id)
    return data, header


def paste_grid_data_into_volume(volume, mcc, experiment_ids, eid, warehouse=None):
    ''' Paste NRRD data into zarr volume at the appropriate experiment index. Thread-safe.

    Parameters
    ----------
    volume : zarr.corr.Array
        Pre-initialized zarr array having grid data dimensions and a 4th dim of len(experiment_ids)
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache
        Used to manage data access
    experiment_ids : list of int
        List of experiment ids used to determine index of eid in array
    eid : int
        The experiment id whose data will be pasted
    warehouse : psycopg2.Connection or None
        Warehouse database connection to use if running with --internal.

    '''

    data, header = get_projection_density_grid_data(eid, mcc, warehouse)
    logging.info('read in data for experiment {0}'.format(eid))

    ii = experiment_ids.index(eid)
    volume[:, :, :, ii] = data
    logging.info('pasted data from experiment {0} ({1})'.format(eid, ii))


def make_projection_volume(experiment_ids, mcc, warehouse=None, tmp_dir=None, max_workers=8):
    ''' Build a 4D array of projection density volumes, with experiment as the 4th axis

    Parameters
    ----------
    experiment_ids : list of int
        Ids of experiments to include
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache
        Used to manage data access
    warehouse : psycopg2.Connection or None
        Warehouse database connection to use if running with --internal.
    tmp_dir : str
        Path to directory in which to build the array as a zarr directory store, else the array
        will be built in memory as a zarr dict store. The directory store will be automatically
        deleted when the zarr array is de-allocated.
    max_workers : int
        Number of threads to use.

    Returns
    -------
    volume : zarr.corr.Array
        Zarr array shaped like grid data of the resolution passed to mcc followed by a
        fourth dimension of size len(experiment_ids), fully populated.

    '''

    data, _ = get_projection_density_grid_data(experiment_ids[0], mcc, warehouse)
    volume_shape = data.shape + (len(experiment_ids),)
    chunks = volume_shape[:-1]+(1,)
    if tmp_dir is not None:
        store = zarr.storage.TempStore(prefix='volume_', dir=tmp_dir)
        volume = zarr.creation.create(shape=volume_shape, dtype=np.float32, chunks=chunks, store=store)
    else:
        volume = zarr.creation.create(shape=volume_shape, dtype=np.float32, chunks=chunks)

    logging.info('volume occupies {0} bytes ({1})'.format(volume.nbytes, volume.dtype.name))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        paste_experiment = partial(paste_grid_data_into_volume, volume, mcc, experiment_ids, warehouse=warehouse)
        for _ in executor.map(paste_experiment, experiment_ids):
            pass

    return volume


def make_injection_structures_arrays(experiments_ds, ontology_depth, structure_paths):
    ''' Build experimentwise arrays of injection structures

    Parameters
    ----------
    experiments_ds : xarray.DataSet
        Contains information about experiments. Must have an attribute injection_structures whose values are / seperated
        strings of experimentwise injection structures.
    ontology_depth : int
        Maximum node depth within the ontology tree.
    structure_paths : dict | int -> list of int
        Maps structure ids to lists of int. Each such list describes the path from the root of the structure tree to
        structure in question.

    Returns
    -------
    injection_structures_arr : numpy.ndarray
        Ragged array of experimentwise injection structures.
    injection_structures_path : numpy.ndarray
        As injection_structures_arr, but with an additional dimension holding the full path to root of each injection
        structure.

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

    Parameters
    ----------
    rgb : list of int
        Red, Green, and Blue values. Should be integer types in the range [0, 255].

    Returns
    -------
    str :
        6-character hex string corresponding to input RGB values

    '''

    hx = ''.join( hex(ii)[2:] if ii >= 16 else '0' + hex(ii)[2:] for ii in rgb )
    return hx


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


def generate_injection_mask(ds, projection_mask, experiment_index):
    ''' Generate injection mask at the given experiment_index and paste into a large zarr array. Thread-safe.

    Parameters
    ----------
    ds : xr.Dataset
        Incoming dataset (not modified by this function).
    projection_mask : zarr.core.Array
        Boolean zarr array shaped like ds.projection.
    experiment_index : int
        Positional index on the experiment dimension for which to generate and paste the mask.

    '''

    logging.info('generating projection mask for experiment {} of {}'.format(experiment_index+1, ds.dims['experiment']))
    injection_structures = ds.injection_structures_array.isel(experiment=experiment_index)
    injection_structures = injection_structures.where(injection_structures!=0, drop=True)
    projection_mask[:,:,:,experiment_index] = (ds.ccf_structures!=injection_structures).all(dim=['depth','secondary'])


def build_projection_mask(ds, tmp_dir=None, max_workers=8):
    ''' Given a dataset with 'projection,' 'injection_structures_array,' and 'ccf_structures' variables,
    generate a boolean DataArray masking elements of 'projection' whose structure annotation belongs
    (ontologically) to one of the corresponding experiment's injection structures.

    ds : xr.Dataset
        Incoming dataset (not modified by this function).
    tmp_dir : str, optional
        Array will be built in a temporary zarr store at this path; else in-memory zarr (DictStore).
    max_workers : int, optional
        Number of threads to use when computing and writing the mask.

    Returns
    -------
    is_projection : xarray.DataArray
        Calculated projection mask as a DataArray which aligns to 'ds'.
    projection_mask : zarr.core.Array
        Projection mask as bare zarr Array, allowing bypass of xarray.

    '''

    chunks = ds.projection.shape[:-1]+(1,)
    if tmp_dir is not None:
        store = zarr.storage.TempStore(prefix='injection_mask_', dir=tmp_dir)
        projection_mask = zarr.creation.create(shape=ds.projection.shape, dtype=np.bool, chunks=chunks, store=store)
    else:
        projection_mask = zarr.creation.create(shape=ds.projection.shape, dtype=np.bool, chunks=chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(partial(generate_injection_mask, ds, projection_mask), range(ds.dims['experiment'])):
            pass

    is_projection = xr.DataArray(da.from_array(projection_mask, chunks=projection_mask.chunks),
        dims=ds.projection.dims, coords=ds.projection.coords)
    return is_projection, projection_mask


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
    ccf_anno_paths = make_annotation_volume_paths(ccf_anno, ontology_depth, structure_paths)

    pv = make_unionize_tables('projection_volume', mcc, all_unionizes_path, experiment_ids, structure_ids, warehouse=warehouse)
    npv = make_unionize_tables('normalized_projection_volume', mcc, all_unionizes_path, experiment_ids, structure_ids, warehouse=warehouse)

    projection_unionize = xr.concat([pv,npv],xr.DataArray([False,True],dims=['normalized'],name='normalized'))
    structure_volumes = make_unionize_tables('volume', mcc, all_unionizes_path, experiment_ids, structure_ids, warehouse=warehouse)

    structure_paths_array = make_structure_paths_array(projection_unionize, ontology_depth, structure_paths)

    injection_structures_arr, injection_structure_paths = make_injection_structures_arrays(
        experiments_ds, ontology_depth, structure_paths)

    primary_structures = experiments_ds.structure_id.values
    primary_structure_paths = make_primary_structure_paths(primary_structures, ontology_depth, structure_paths)

    volume = make_projection_volume(experiment_ids, mcc, warehouse=warehouse, tmp_dir=args.data_dir)

    ccf_dims = ['anterior_posterior', 'superior_inferior', 'left_right']
    ds = xr.Dataset(
        data_vars={
            'ccf_structure': (ccf_dims, ccf_anno, {'spacing': [args.resolution]*3}),
            'ccf_structures': (ccf_dims+['depth'], ccf_anno_paths),
            'projection': (ccf_dims+['experiment'], da.from_array(volume, chunks=volume.chunks)),
            'is_summary_structure': (['structure'], [structure.item() in summary_structures for structure in projection_unionize.structure]),
            'structure_color': structure_colors,
            'volume': projection_unionize,
            'structure_volumes': structure_volumes,
            'primary_structures': (['experiment', 'depth'], primary_structure_paths),
            'injection_structures_array': (['experiment', 'secondary'], injection_structures_arr),
            'injection_structure_paths': (['experiment', 'secondary', 'depth'], injection_structure_paths)
        },
        coords={
            'experiment': experiment_ids,
            'structures': (['structure', 'depth'], structure_paths_array),
            'anterior_posterior': args.resolution*np.arange(ccf_anno.shape[0]),
            'superior_inferior': args.resolution*np.arange(ccf_anno.shape[1]),
            'left_right': args.resolution*np.arange(ccf_anno.shape[2]),
        }
    )

    ds.merge(experiments_ds, inplace=True, join='exact')
    ds.merge(structure_meta, inplace=True, join='left')
    ds['is_primary'] = (ds.structure_id==ds.structures).any(dim='depth') #todo: make it possible to do this masking on-the-fly (?)
    ds['is_projection'], projection_mask = build_projection_mask(ds, tmp_dir=args.data_dir)

    # make flat versions of ccf-shaped fields and drop non-annotated voxels
    ds_flat = ds[['projection', 'is_projection', 'ccf_structure', 'ccf_structures']]
    ds_flat.rename({var: '{}_flat'.format(var) for var in set(ds_flat.variables)-set(['experiment'])}, inplace=True)
    ds_flat = ds_flat.stack(ccf=ds_flat.ccf_structure_flat.dims)
    ds_flat.reset_index('ccf', inplace=True) # multi-index not netcdf compatible
    ds_flat = ds_flat.where(ds_flat.ccf_structure_flat!=0, drop=True)
    # where seems to clobber some dtypes
    ds_flat['projection_flat'] = ds_flat.projection_flat.astype(ds.projection.dtype)
    ds_flat['is_projection_flat'] = ds_flat.is_projection_flat.astype(ds.is_projection.dtype)
    ds_flat['ccf_structure_flat'] = ds_flat.ccf_structure_flat.astype(ds.ccf_structure.dtype)
    ds_flat['ccf_structures_flat'] = ds_flat.ccf_structures_flat.astype(ds.ccf_structures.dtype)
    ds.merge(ds_flat, inplace=True)

    store_file = os.path.join(args.data_dir, args.data_name + '.zarr.lmdb')
    store = zarr.storage.LMDBStore(store_file)
    logging.info('writing dataset to {}'.format(store_file))
    # monkey patch to get zarr to ignore dask chunks and use its own heuristics
    def copy_func(f):
        g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                               argdefs=f.__defaults__,
                               closure=f.__closure__)
        g = functools.update_wrapper(g, f)
        g.__kwdefaults__ = f.__kwdefaults__
        return g
    orig_determine_zarr_chunks = copy_func(xr.backends.zarr._determine_zarr_chunks)
    xr.backends.zarr._determine_zarr_chunks = lambda enc_chunks, var_chunks, ndim: orig_determine_zarr_chunks(enc_chunks, None, ndim)
    # encoding settings; these could be tweaked
    encoding = {
        'is_projection': {'filters': [numcodecs.packbits.PackBits()]},
        'projection': {'filters': [numcodecs.quantize.Quantize(3, np.float32)]}
        }
    compressor = numcodecs.blosc.Blosc(cname='snappy', clevel=1, shuffle=numcodecs.blosc.Blosc.SHUFFLE)
    # monkey patch to make dask arrays writable with different chunks than zarr dest
    # could without this but would have to contend with 'inconsistent chunks' on dataset
    def sync_using_zarr_copy(self, compute=True):
        if self.sources:
            import dask.array as da
            rechunked_sources = [source.rechunk(target.chunks)
                for source, target in zip(self.sources, self.targets)]
            delayed_store = da.store(rechunked_sources, self.targets,
                                     lock=self.lock, compute=compute,
                                     flush=True)
            self.sources = []
            self.targets = []
            return delayed_store
    xr.backends.common.ArrayWriter.sync = sync_using_zarr_copy
    # to_zarr will fail with object columns containing None
    for field in ds.variables:
        if ds[field].dtype.name == 'object':
            ds[field] = ds[field].astype('str')

    # write to zarr with overridable default encoding settings
    ds.to_zarr(
        store=store,
        encoding={var: {**{'chunks': None, 'compressor': compressor}, **encoding.get(var, {})} for var in ds.variables}
        )
    logging.info('wrote dataset to {}'.format(store_file))

    #PBS-1262:
    shutil.copy2(args.surface_coords_path, args.data_dir)


if __name__ == '__main__':

    logging.getLogger('').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Connectivity datacube generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mouse_ccf', help="base name with which to create files")
    parser.add_argument('--manifest_filename', type=str, default='mouse_connectivity_manifest.json')
    parser.add_argument('--resolution', type=int, default=100, help='isometric CCF resolution')
    parser.add_argument('--surface_coords_path', type=str, default=DEFAULT_SURFACE_COORDS_PATH)
    parser.add_argument('--internal', action='store_true')

    args = parser.parse_args()

    warehouse = None
    if args.internal:
        # NOTE: this connection is used by multiple concurrent threads
        warehouse_key = urllib.parse.urlparse(args.data_src).netloc
        warehouse = psycopg2.connect(**WAREHOUSE_DATABASES[warehouse_key])

    main()
