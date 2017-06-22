#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import re
import urllib
import argparse
from util.convert import pd_dataframe_to_np_structured_array
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi


def main():
    parser = argparse.ArgumentParser(description='ApiCamCellMetric pandas data generator script')
    parser.add_argument('--data-src', default='http://api.brain-map.org/', help='base RMA url from which to load data')
    parser.add_argument('--data-dir', default='./', help='load CSV and NPY files from this directory')
    args = parser.parse_args()

    generate(args.data_src, args.data_dir)


def generate(data_src=None, data_dir='./'):
    print('Generating...')
    csv_file = data_dir + 'cell_specimens.csv'

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
    df.to_csv(csv_file)

    npyfile = re.sub('\.csv', '.npy', csv_file, flags=re.I)
    df = pd.read_csv(csv_file)
    df.to_csv(csv_file, index=('index' not in list(df)), index_label='index')
    sa = pd_dataframe_to_np_structured_array(df)
    del df
    np.save(npyfile, sa)
    del sa
    print('Data created in data_dir.')


if __name__ == '__main__':
    main()

