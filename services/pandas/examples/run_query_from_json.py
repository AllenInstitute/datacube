import argparse
import json
import sys

import requests


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


def produce_output(result, output_path):

    if output_path is not None:
        with open(output_path, 'w') as output_file:
            json.dump(result, output_file, indent=2)
    else:
        json.dump(result, sys.stdout, indent=2)


def clean_host(host):
    if not 'http' in host:
        host = 'http://{}'.format(host)
    return host


def main(query_path, host, port, output_path):

    query = load_query_from_json(query_path)

    host = clean_host(host)
    result = post_datacube_request(query, host, port)

    produce_output(result, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', type=str)
    parser.add_argument('host', type=str)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()
    main(args.query_path, args.host, args.port, args.output_path)