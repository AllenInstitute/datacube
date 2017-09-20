#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import re
import urllib
import argparse
from util import pd_dataframe_to_np_structured_array, np_structured_array_to_xr_dataset
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi


def main():
    parser = argparse.ArgumentParser(description='ApiCamCellMetric pandas data generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='save CSV and NPY files in this directory')
    parser.add_argument('--data-name', default='cell_specimens', help="base name with which to create files")
    args = parser.parse_args()

    generate(args.data_src, args.data_dir, args.data_name)


def generate(data_src=None, data_dir='./', data_name='cell_specimens'):
    print('Generating...')
    csv_file = data_dir + data_name + '.csv'

    # download directly over SQL
    #import sqlalchemy as sa
    #sql = 'select * from api_cam_cell_metrics'
    #con = sa.create_engine(data_src) # data_src = postgresql://user:pass@host:port/dbname
    #df = pd.read_sql(sql, con)
    #if not os.path.exists(data_dir):
    #    os.makedirs(data_dir)
    #df.to_csv(csv_file)

    # manually download over RMA
    #csv_url = data_src + '/api/v2/data/ApiCamCellMetric/query.csv?num_rows=all'
    #if not os.path.exists(data_dir):
    #    os.makedirs(data_dir)
    #urllib.urlretrieve(csv_url, csv_file)
    #df = pd.read_csv(csv_file, true_values=['t'], false_values=['f'])
    #df.to_csv(csv_file)

    # use SDK paged RMA download
    api = BrainObservatoryApi(base_uri=data_src)
    data = api.get_cell_metrics()
    df = pd.DataFrame.from_dict(data)

    nc_file = re.sub('\.csv', '.nc', csv_file, flags=re.I)
    npy_file = re.sub('\.csv', '.npy', csv_file, flags=re.I)
    sa = pd_dataframe_to_np_structured_array(df)
    np.save(npy_file, sa)
    del df
    ds = np_structured_array_to_xr_dataset(sa)
    del sa
    ds.to_netcdf(nc_file, format='NETCDF4')
    print('Data created in data_dir.')


if __name__ == '__main__':
    main()

