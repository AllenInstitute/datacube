import argparse
import os
import json
import io

import requests
import pandas as pd


def file_directory():
    return os.path.dirname(__file__)
file_directory = file_directory()


def download_csv(query_string, file_path):
    response = requests.get(query_string)
    dataframe = pd.read_csv(io.StringIO(response.text))
    dataframe.to_csv(file_path, index=False)


def download_json(query_string, file_path):
    response = requests.get(query_string)
    with open(file_path, 'w') as results_file:
        json.dump(response.json(), results_file, indent=2)


def safe_make_parent_dirs(path):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


requester_dispatch = {
    'csv': download_csv,
    'json': download_json
} # maps file extensions to an appropriate function for downloading files of that type


def main(manifest_path, data_dir, base, force):

    with open(manifest_path, 'r') as manifest_file:
        manifest = json.load(manifest_file)

    for query in manifest:
        results_filename = '{name}.{extension}'.format(**query)
        results_path = os.path.join(data_dir, results_filename)
        
        datacube_filename = '{name}_echo.{extension}'.format(**query)
        datacube_path = os.path.join(data_dir, datacube_filename)
        datacube_query = '&'.join([query['query'], 'echo=true'])

        if os.path.exists(results_path) and not force:
            print('skipping {}'.format(query['name']))
            continue

        if os.path.exists(results_path) and not force:
            print('skipping {}'.format(query['name']))
            continue

        requester = requester_dispatch[query['extension']]
        safe_make_parent_dirs(results_path)

        if base is None:
            base = query['base']

        requester('/'.join([base, query['query']]), results_path)
        print('wrote query response for {}'.format(query['name']))

        requester('/'.join([base, datacube_query]), datacube_path)
        print('wrote echo response for {}'.format(query['name']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a set of queries from a manifest file and stores the results locally. ')
    parser.add_argument('--manifest_path', type=str, default=os.path.join(file_directory, 'manifest.json'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(file_directory, 'data'))
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--force', action='store_true', default=False)

    args = parser.parse_args()
    main(args.manifest_path, args.data_dir, args.base, args.force)