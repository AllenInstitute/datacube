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

    registration = requests.post(
        router_uri, 
        json={
            "procedure": "wamp.registration.list",
            "args": [],
            "kwargs": {}
        }
    ).json()

    registration_checker = functools.partial(check_registration, router_uri)
    codomain = registration['args'][0][u'prefix'] \
        + registration['args'][0][u'exact'] \
        + registration['args'][0][u'wildcard']

    return map(registration_checker, codomain)


def format_output(output):
    print("Content-Type: text/html\n\n")

    print("""<!doctype html>
    <html>
        <body>
            {}
        </body>
    </html>
    """.format(s))



def main(router_uri):

    wamp_results = check_registration_list(router_uri)

    s=''
    for proc_uri in [ret['args'][0]['uri'] for ret in r]:
        if 'status' in proc_uri:
            try:
                ret=requests.post(router_uri, json={"procedure": proc_uri}, timeout=10.).json()
                s += '<p>{}: {}</p>'.format(proc_uri, 'up' if ret['args'][0] == True else 'down')
            except:
                s += '<p>{}: down</p>'.format(proc_uri)

    format_output(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--router_uri', type=str, default='http://localhost:8080/call')

    args = parser.parse_args()
    main(args.router_uri)






