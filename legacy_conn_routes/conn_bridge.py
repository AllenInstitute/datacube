import sys
import argparse
import functools

import simplejson
from klein import Klein
from autobahn.twisted.wamp import Application
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.web.server import Site
from twisted.internet import reactor
from twisted.python import log

from legacy_routes.queries.search_differential_rows import get_structure_search_kwargs, postprocess_search_differential_rows
from legacy_routes.queries.correlation_search import get_correlation_search_kwargs, postprocess_correlation_search
from legacy_routes.queries.spatial_search import get_spatial_search_kwargs, postprocess_spatial_search
from legacy_routes.queries.injection_coordinate import get_injection_coordinate_kwargs, postprocess_injection_coordinates_search

from legacy_routes.utilities.request import decode
from legacy_routes.utilities.server import get_open_tcp_port
from legacy_routes.utilities.response import add_json_headers, add_csv_headers, reorient_datacube_response, package_json_response


app = Klein()
wampapp = Application()


def process_request(request):

    add_json_headers(request)
    args = decode(request.args)
    echo = args.pop('echo', False)

    return args, echo


def call_datacube(procedure, posargs, kwargs, echo=False):
    if echo:
        returnValue(simplejson.dumps({'procedure': procedure, 'args': posargs, 'kwargs': kwargs}))
    return wampapp.session.call(procedure, *posargs, **kwargs)


@app.route('/data/search/injection_coordinates', methods=('GET',))
@inlineCallbacks
def mouseconn_coordinate(request):

    args, echo = process_request(request)

    res = yield call_datacube(
        'org.brain-map.api.datacube.raw.connectivity',
        [],
        get_injection_coordinate_kwargs(**args),
        echo
    )

    res = reorient_datacube_response(
        res, 
        process_df=functools.partial(
            postprocess_injection_coordinates_search,
            seed=args['seedPoint'],
            showDetail=args.get('showDetail', False)
        )
    )

    res = package_json_response(res)
    returnValue(simplejson.dumps(res))



@app.route('/data/projection_map/target', methods=('GET',))
@inlineCallbacks
def mouseconn_spatial(request):

    args, echo = process_request(request)
    
    res = yield call_datacube(
        'org.brain-map.api.datacube.conn_spatial_search',
        [{
            'anterior_posterior': args['seedPoint'][0],
            'superior_inferior': args['seedPoint'][1],
            'left_right': args['seedPoint'][2]
        }], 
        get_spatial_search_kwargs(**args), 
        echo
    )

    res = reorient_datacube_response(res, process_df=postprocess_spatial_search)

    res = package_json_response(res)
    returnValue(simplejson.dumps(res))



@app.route('/data/search/injection_rows', methods=('GET',))
@inlineCallbacks
def mouseconn_structure(request):
    
    args, echo = process_request(request)

    res = yield call_datacube(
        'org.brain-map.api.datacube.raw.connectivity',
        [], 
        get_structure_search_kwargs(**args), 
        echo
    )

    res = reorient_datacube_response(
        res, 
        process_df=functools.partial(
            postprocess_search_differential_rows, 
            showDetail=args.get('showDetail', False)
        )
    )

    res = package_json_response(res)
    returnValue(simplejson.dumps(res))


@app.route('/data/search/correlated_rows', methods=('GET',))
@inlineCallbacks
def mouseconn_correlation(request):

    args, echo = process_request(request)

    res = yield call_datacube(
        'org.brain-map.api.datacube.corr.connectivity', 
        ['projection_flat', 'experiment', args['seed']], 
        get_correlation_search_kwargs(**args), 
        echo
    )

    res = reorient_datacube_response(
        res, 
        process_df=functools.partial(
            postprocess_correlation_search, 
            showDetail=args.get('showDetail', False)
        )
    )

    res = package_json_response(res)
    returnValue(simplejson.dumps(res))


@app.route('/data/health', methods=('GET',))
@inlineCallbacks
def health(request):
    res = yield {}
    res = package_json_response(res)
    returnValue(simplejson.dumps(res))


def main():

    reactor.listenTCP(args.port, Site(app.resource()))
    wampapp.run(args.wamp_transport, args.wamp_realm)


if __name__ == '__main__':
    log.startLogging(sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wamp_transport', type=str, default='ws://tdatacube:8080/ws')
    parser.add_argument('--wamp_realm', type=str, default='aibs')
    parser.add_argument('--port', type=int, default=get_open_tcp_port())

    args = parser.parse_args()
    main()