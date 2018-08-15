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

    chunksize = 1000
    logging.info('reading exon-matrix csv')
    exon_expression = np.empty((len(df_rows), len(df_cols)), dtype=np.int32)
    for chunk in pd.read_csv(args.data_dir + 'human_MTG_2018-06-14_exon-matrix.csv', dtype=np.int32, chunksize=chunksize):
        exon_expression[chunk.index[0]:(chunk.index[-1]+1),:] = chunk.values[:,1:]
    logging.info('reading intron-matrix csv')
    intron_expression = np.empty((len(df_rows), len(df_cols)), dtype=np.int32)
    for chunk in pd.read_csv(args.data_dir + 'human_MTG_2018-06-14_intron-matrix.csv', dtype=np.int32, chunksize=chunksize):
        intron_expression[chunk.index[0]:(chunk.index[-1]+1),:] = chunk.values[:,1:]

    logging.info('merging into dataset')
    ds_rows = xr.Dataset.from_dataframe(df_rows).rename({'gene': 'gene_symbol', 'index': 'gene'})
    ds_rows.set_index(gene='entrez_id', inplace=True)
    ds_cols = xr.Dataset.from_dataframe(df_cols).rename({'index': 'nucleus'})
    ds_cols.set_index(nucleus='sample_id', inplace=True)
    ds = xr.merge([ds_rows, ds_cols])
    ds['exon_expression'] = xr.DataArray(exon_expression, dims=('gene', 'nucleus'))
    ds['intron_expression'] = xr.DataArray(intron_expression, dims=('gene', 'nucleus'))

    store_file = os.path.join(args.data_dir, args.data_name + '.zarr.lmdb')
    store = zarr.storage.LMDBStore(store_file)
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
