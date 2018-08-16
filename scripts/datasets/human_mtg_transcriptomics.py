#!/usr/bin/env python

from __future__ import division
import argparse
import os
import logging

import wget
import zipfile

import numpy as np
import xarray as xr
import pandas as pd
import zarr
import dask
import dask.array as da


def read_csv(filename, tmp_dir, dims, nrows, ncols, dtype=None, chunksize=1000):
    store = zarr.storage.TempStore(prefix='read_csv_', dir=tmp_dir)
    data = zarr.creation.create(shape=(nrows, ncols), dtype=dtype, chunks=(chunksize, ncols), store=store)
    for chunk in pd.read_csv(args.data_dir + filename, dtype=dtype, chunksize=chunksize):
        data[chunk.index[0]:(chunk.index[-1]+1),:] = chunk.values[:,1:]
    result = xr.DataArray(da.from_array(data, chunks=data.chunks), dims=dims)
    return result


def main():
    url = args.data_src + '/api/v2/well_known_file_download/694416044'
    logging.info('downloading matrix zipfile from {} to {}'.format(url, args.data_dir))
    filename = wget.download(url, out=args.data_dir, bar=lambda *args: None)

    logging.info('unzipping {} to {}'.format(filename, args.data_dir))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(args.data_dir)
    zip_ref.close()

    logging.info('reading rows csv')
    df_rows = pd.read_csv(args.data_dir + 'human_MTG_2018-06-14_genes-rows.csv')
    logging.info('reading cols csv')
    df_cols = pd.read_csv(args.data_dir + 'human_MTG_2018-06-14_samples-columns.csv')

    logging.info('reading exon-matrix csv')
    exon_expression = read_csv('human_MTG_2018-06-14_exon-matrix.csv', args.data_dir, ('gene', 'nucleus'), len(df_rows), len(df_cols), dtype=np.int32)
    logging.info('reading intron-matrix csv')
    intron_expression = read_csv( 'human_MTG_2018-06-14_intron-matrix.csv', args.data_dir, ('gene', 'nucleus'), len(df_rows), len(df_cols), dtype=np.int32)

    logging.info('merging into dataset')
    ds_rows = xr.Dataset.from_dataframe(df_rows).rename({'gene': 'gene_symbol', 'index': 'gene'})
    ds_rows.set_index(gene='entrez_id', inplace=True)
    ds_cols = xr.Dataset.from_dataframe(df_cols).rename({'index': 'nucleus'})
    ds_cols.set_index(nucleus='sample_id', inplace=True)
    ds = xr.merge([ds_rows, ds_cols])
    ds['exon_expression'] = exon_expression
    ds['intron_expression'] = intron_expression

    store_file = os.path.join(args.data_dir, args.data_name + '.zarr.lmdb')
    store = zarr.storage.NestedDirectoryStore(store_file)
    logging.info('writing dataset to {}'.format(store_file))
    ds.to_zarr(store=store)
    logging.info('wrote dataset to {}'.format(store_file))


if __name__ == '__main__':

    logging.getLogger('').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Connectivity datacube generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save file(s) in this directory')
    parser.add_argument('--data-name', default='mouse_ccf', help="base name with which to create files")
    #parser.add_argument('--internal', action='store_true')

    args = parser.parse_args()

    main()
