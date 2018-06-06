import argparse
import json
import sys

import requests

import xarray as xr


def load_query_from_json(query_path):

    with open(query_path) as query_file:
        query = json.load(query_file)

    if 'query' in query:
        query = query['query']

    return query


def post_datacube_request(query, host, port):

    response = requests.post(
        '{}:{}/call'.format(host, port), 
        data=json.dumps(query), 
        headers={'Content-Type': 'application/json'}
    )

    return response.json()


def produce_output(result, output_path, output_format):

    if output_path is not None:
        with open(output_path, 'w') as output_file:
            json.dump(result, output_file, indent=2)
    else:
        if output_format == 'xarray':
            try:
                print(xr.Dataset.from_dict(result['args'][0]))
                return
            except:
                pass
        json.dump(result, sys.stdout, indent=2)


def clean_host(host):
    if not 'http' in host:
        host = 'http://{}'.format(host)
    return host


def main(query_path, host, port, output_path, output_format):

    query = load_query_from_json(query_path)

    host = clean_host(host)
    result = post_datacube_request(query, host, port)

    produce_output(result, output_path, output_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', type=str, help='path to json file containing query. The file may contain the query as is, or as the value associated with a top-level \"query\" key.')
    parser.add_argument('host', type=str, help='Datacube host. If \"http\" is not in the hostname, it will be prepended.')
    parser.add_argument('--port', type=int, default=8080, help='Datacube port. Defaults to 8080.')
    parser.add_argument('--output_path', type=str, default=None, help='If provided, outputs will be written to this file. Otherwise, they will be written to stdout.')
    parser.add_argument('--output_format', type=str, default='json', help='Output format, defaults to json. If \'xarray\', will attempt to interpret json as xarray dataset.')

    args = parser.parse_args()
    main(args.query_path, args.host, args.port, args.output_path, args.output_format)
