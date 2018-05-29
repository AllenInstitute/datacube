import sys
import os
import requests
import json
import argparse
import functools


def check_registration(router_uri, reg_id):
    return requests.post(
        router_uri,
        json={"procedure": "wamp.registration.get", "args": [reg_id]}
    ).json()


def check_registration_list(router_uri):

    uri = '{}/call'.format(router_uri)

    registration = requests.post(
        uri, 
        json={
            "procedure": "wamp.registration.list",
            "args": [],
            "kwargs": {}
        }
    ).json()

    registration_checker = functools.partial(check_registration, uri)
    codomain = registration['args'][0][u'prefix'] \
        + registration['args'][0][u'exact'] \
        + registration['args'][0][u'wildcard']
    registered_procedures = map(registration_checker, codomain)

    up = {}
    for proc_uri in [ret['args'][0]['uri'] for ret in registered_procedures]:
        if 'status' in proc_uri:
            try:
                ret = requests.post(uri, json={"procedure": proc_uri}, timeout=10.).json()
                up[proc_uri] = True if ret['args'][0] == True else False
            except:
                up[proc_uri] = False

    return up


def check_conn_bridge(router_uri):

    uri = '{}/mouseconn/data/health'.format(router_uri)
    
    results = {}
    try:
        response = requests.get(uri).json()
    except Exception as err:
        results['conn_bridge'] = False
        return results

    if 'success' in response and response['success'] == 'true':
        results['conn_bridge'] = True
    else:
        results['conn_bridge'] = False

    print(response)

    return results


def format_output(output):

    get_state_str = lambda state: 'up' if state else 'down'
    output = [ '<p>{}: {}</p>'.format(key, get_state_str(value)) for key, value in output.items() ]

    print("Content-Type: text/html\n\n")

    print("""<!doctype html>
    <html>
        <body>
            {}
        </body>
    </html>
    """.format(''.join(output)))



def main(router_uri):

    wamp_results = check_registration_list(router_uri)
    wamp_results.update(check_conn_bridge(router_uri))
    format_output(wamp_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--router_uri', type=str, default='http://localhost:8080')

    args = parser.parse_args()
    main(args.router_uri)






