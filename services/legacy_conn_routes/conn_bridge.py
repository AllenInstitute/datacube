import sys
import argparse
import functools

import simplejson
from klein import Klein
from autobahn.twisted.wamp import ApplicationRunner, ApplicationSession
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.web.server import Site
from twisted.internet import reactor
from twisted.python import log

from legacy_routes.clauses.domain import DEFAULT_DOMAIN_THRESHOLD   

from legacy_routes.queries.search_differential_rows import get_structure_search_kwargs, postprocess_search_differential_rows, handle_domain_threshold_args
from legacy_routes.queries.correlation_search import get_correlation_search_kwargs, postprocess_correlation_search
from legacy_routes.queries.spatial_search import get_spatial_search_kwargs, postprocess_spatial_search
from legacy_routes.queries.injection_coordinate import get_injection_coordinate_kwargs, postprocess_injection_coordinates_search

from legacy_routes.utilities.request import decode
from legacy_routes.utilities.server import get_open_tcp_port
from legacy_routes.utilities.response import add_json_headers, add_csv_headers, reorient_datacube_response, package_json_response
from legacy_routes.utilities.ccf_data_store import CcfDataStore


session = [None]


class GlobalSessionComponent(ApplicationSession):
    def onJoin(self, details):
        session[0] = self


def process_request(request):

    add_json_headers(request)
    args = decode(request.args)
    echo = args.pop('echo', False)

    return args, echo


def call_datacube(procedure, posargs, kwargs, echo=False):
    if echo:
        returnValue(simplejson.dumps({'procedure': procedure, 'args': posargs, 'kwargs': kwargs}))
    return session[0].call(procedure, *posargs, **kwargs)


class ConnBridgeApp(object):
    app = Klein()

    def __init__(self, ccf_store):
        self.ccf_store = ccf_store


    @app.handle_errors(Exception)
    def log_request_and_failure(self, request, failure):
        log.err('Request {} failed with {}.'.format(str(request), str(failure)))


    @app.route('/data/P56/search/injection_coordinates', methods=('GET',))
    @app.route('/data/search/injection_coordinates', methods=('GET',))
    @inlineCallbacks
    def mouseconn_coordinate(self, request):

        args, echo = process_request(request)

        res = yield call_datacube(
            'org.brain-map.api.datacube.raw.connectivity',
            [],
            get_injection_coordinate_kwargs(acronym_id_map=self.ccf_store.acronym_id_map, **args),
            echo
        )

        res = reorient_datacube_response(
            res, 
            process_df=functools.partial(
                postprocess_injection_coordinates_search,
                seed=args['seedPoint'],
                showDetail=args.get('showDetail', False),
                ccf_store=self.ccf_store
            )
        )

        res = package_json_response(res)
        returnValue(simplejson.dumps(res))


    @app.route('/data/P56/projection_map/target', methods=('GET',))
    @app.route('/data/projection_map/target', methods=('GET',))
    @inlineCallbacks
    def mouseconn_spatial(self, request):

        args, echo = process_request(request)
        
        res = yield call_datacube(
            'org.brain-map.api.datacube.conn_spatial_search',
            [{
                'anterior_posterior': args['seedPoint'][0],
                'superior_inferior': args['seedPoint'][1],
                'left_right': args['seedPoint'][2]
            }], 
            get_spatial_search_kwargs(acronym_id_map=self.ccf_store.acronym_id_map, **args), 
            echo
        )

        res = reorient_datacube_response(res, process_df=functools.partial(
            postprocess_spatial_search,
            ccf_store=self.ccf_store
        ))

        res = package_json_response(res)
        returnValue(simplejson.dumps(res))

    @app.route('/data/P56/search/injection_rows', methods=('GET',))
    @app.route('/data/search/injection_rows', methods=('GET',))
    @inlineCallbacks
    def mouseconn_structure(self, request):
        
        args, echo = process_request(request)
        handle_domain_threshold_args(args, 'injection')
        handle_domain_threshold_args(args, 'target')

        res = yield call_datacube(
            'org.brain-map.api.datacube.raw.connectivity',
            [], 
            get_structure_search_kwargs(acronym_id_map=self.ccf_store.acronym_id_map, **args), 
            echo
        )

        res = reorient_datacube_response(
            res, 
            process_df=functools.partial(
                postprocess_search_differential_rows, 
                showDetail=args.get('showDetail', False),
                sum_threshold=args.get('target_domain_threshold', DEFAULT_DOMAIN_THRESHOLD),
                ccf_store=self.ccf_store
            )
        )

        res = package_json_response(res)
        returnValue(simplejson.dumps(res))


    @app.route('/data/P56/search/correlated_rows', methods=('GET',))
    @app.route('/data/search/correlated_rows', methods=('GET',))
    @inlineCallbacks
    def mouseconn_correlation(self, request):

        args, echo = process_request(request)

        res = yield call_datacube(
            'org.brain-map.api.datacube.corr.connectivity', 
            ['projection_flat', 'experiment', args['seed']], 
            get_correlation_search_kwargs(acronym_id_map=self.ccf_store.acronym_id_map, **args), 
            echo
        )

        res = reorient_datacube_response(
            res, 
            process_df=functools.partial(
                postprocess_correlation_search, 
                showDetail=args.get('showDetail', False),
                ccf_store=self.ccf_store
            )
        )

        res = package_json_response(res)
        returnValue(simplejson.dumps(res))


    @app.route('/data/health', methods=('GET',))
    def health(self, request):
        res = package_json_response({})
        return simplejson.dumps(res)



class ConnBridgeResourceFactory:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wamp_transport', type=str, default='ws://tdatacube:8080/ws')
    parser.add_argument('--wamp_realm', type=str, default='aibs')
    parser.add_argument('--port', type=int, default=get_open_tcp_port())
    parser.add_argument('--ontology_path', type=str, default=None)

    def resource(args, start=False):
        args = ConnBridgeResourceFactory.parser.parse_args(args)

        ccf_store = None
        if args.ontology_path is not None:
            ccf_store = CcfDataStore(args.ontology_path)

        cba = ConnBridgeApp(ccf_store)

        runner = ApplicationRunner(str(args.wamp_transport), str(args.wamp_realm))
        runner.run(GlobalSessionComponent, auto_reconnect=True, start_reactor=False)

        if start:
            reactor.listenTCP(args.port, Site(cba.app.resource()))
            print('conn_bridge running on localhost:{}'.format(args.port))
            reactor.run()
        else:
            return cba.app.resource()

resource = ConnBridgeResourceFactory.resource


def main():
    log.startLogging(sys.stdout)
    cbr = ConnBridgeResourceFactory.resource(sys.argv[1:], start=True)

if __name__ == '__main__':
    main()
