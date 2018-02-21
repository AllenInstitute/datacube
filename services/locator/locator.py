#!/usr/bin/env python

import sys
import os
import txaio
import argparse

from os import environ
from twisted.internet import threads
from twisted.internet.defer import inlineCallbacks
#from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from wamp import ApplicationSession, ApplicationRunner # copy of stock wamp.py with modified timeouts
from twisted.internet.defer import inlineCallbacks, returnValue
from autobahn.wamp.auth import compute_wcs

from configuration_manager import ConfigurationManager

from classes.surface_projection import SurfacePoint
from classes.projection_point import ProjectionPoint
from classes.filmstrip_locator import FilmStripLocator
from classes.model_loader import ModelLoader
from classes.voxel_lookup import VoxelLookup
from classes.line_finder import LineFinder
from classes.spatial_search import SpatialSearch
from classes.ontology_service import OntologyService
from classes.model_loader import ModelLoader




#  ],
# "components": [
#       {
#          "type": "class",
#          "classname": "locator.locator.LocatorServiceComponent",
#          "realm": "aibs",
#          "role": "authenticated"
#      }
#  ],
#  "options":{
#      "pythonpath": ["../services/"]
#  }

class LocatorServiceComponent(ApplicationSession):

    def onConnect(self):
        self.join(str(args.realm), [u'wampcra'], str(args.username))


    def onChallenge(self, challenge):
        if challenge.method == u'wampcra':
            signature = compute_wcs(str(args.password).encode('utf8'), challenge.extra['challenge'].encode('utf8'))
            return signature.decode('ascii')

    @inlineCallbacks
    def onJoin(self, details):
        global ready
        
        try:
            path = args.env_vars_path
            config = ConfigurationManager(path)
        except (IOError) as e:
            print(e)


        ####################################################################
        ########################## Endpoints ###############################
        ####################################################################

        def describe ():
            result = 'Locator Service \n\n ------------------------ \n\n'
            result += 'get_surface_point([x,y,z]) => [x,y,z] \n' 

            return result


        @inlineCallbacks
        def surface_point (seedPoint = None):
            results = dict()
            results.setdefault("success", False)

            # Validate Arguments
            if seedPoint == None:
                status = results.setdefault("status", dict())
                status.setdefault("message", "requires seed point")
                
                returnValue(results)
            
            try:
                projector = SurfacePoint(config)
                coordinate = yield threads.deferToThread(projector.get, seedPoint)
                results.setdefault("coordinate", coordinate)
                results['success'] = True

            except (IOError) as e:
                results.setdefault("message", str(e))

            returnValue(results)


        @inlineCallbacks
        def projection_point (path = None, pixel = None):
            results = dict()
            results.setdefault("success", False)

            # Validate Arguments
            if path == None or pixel == None or len(pixel) < 2:
                status = results.setdefault("status", dict())
                status.setdefault("message", "missing path or pixel argument(s)")

                returnValue(results)

            try:
                projector = ProjectionPoint(config)
                coordinate = yield threads.deferToThread(projector.get, path, pixel)
                results.setdefault("coordinate", coordinate)
                results["success"] = True

            except (IOError, ValueError) as e:
                results.setdefault("message", str(e))

            returnValue(results)


        @inlineCallbacks
        def filmstrip_location (pixel = None, distanceMapPath = None, direction = None):
            results = dict()
            results.setdefault("success", False)

            # Validate Arguments
            if pixel == None or distanceMapPath == None or direction == None:
                status = results.setdefault("status", dict())
                status.setdefault("message", "missing pixel, distanceMapPath, or direction argument(s)")


            locator = FilmStripLocator(config)
            results = yield threads.deferToThread(locator.get, pixel, distanceMapPath, direction, results)
                
            returnValue(results)


        @inlineCallbacks
        def voxel_lookup (coords = None):
            results = dict()
            results.setdefault("success", False)
            
            lookup = VoxelLookup(config)
            
            results = yield threads.deferToThread(lookup.get, coords, results)

            returnValue(results)


        # Retrieves streamlines or neuron reconstructions by an ambiguous id
        @inlineCallbacks
        def get_lines (id = None):
            results = dict()
            results.setdefault("success", False)

            locator = LineFinder(config)

            results = yield threads.deferToThread(locator.get, id, results)
            
            returnValue(results)


        @inlineCallbacks
        def spatial_search(voxel = None, map_dir = None):

            search = SpatialSearch(config)

            results = yield threads.deferToThread(search.get, voxel, map_dir)

            returnValue(results)


        @inlineCallbacks
        def ccf_ontology():
            service = OntologyService(config)

            results = yield threads.deferToThread(service.get_ontology)
            returnValue(results)


        @inlineCallbacks        
        def ccf_model(id = None):
            service = ModelLoader(config)
            
            results = yield threads.deferToThread(service.get, id)
            returnValue(results)


        ready = False
        try:
            ####################################################################
            ###################### Endpoint Registration #######################
            ####################################################################

            yield self.register(lambda: True,       u"org.brain_map.locator.status")
            yield self.register(describe,           u"org.brain_map.locator.describe")
            yield self.register(surface_point,      u"org.brain_map.locator.get_surface_point")
            yield self.register(projection_point,   u"org.brain_map.locator.get_projection_point")
            yield self.register(filmstrip_location, u"org.brain_map.locator.get_filmstrip_location")
            yield self.register(voxel_lookup,       u"org.brain_map.locator.get_voxel_structure")
            yield self.register(get_lines,          u"org.brain_map.locator.get_lines")
            yield self.register(spatial_search,     u"org.brain_map.locator.get_streamlines_at_voxel")
            yield self.register(ccf_ontology,       u"org.brain_map.locator.get_ccf_ontology")
            yield self.register(ccf_model,          u"org.brain_map.locator.get_ccf_model")
        
            ready = True
        except (Exception) as e:
            print("Could not register procedure: {0}".format(e))

        if ready:
            print ("Locator Ready!")
        else:
            print("Could not ready Locator Component")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Locator Service')
    parser.add_argument('router', help='url of WAMP router to connect to e.g. ws://localhost:9000/ws')
    parser.add_argument('realm', help='WAMP realm name to join')
    parser.add_argument('username', help='WAMP-CRA username')
    parser.add_argument('password', help='WAMP-CRA secret')
    parser.add_argument('env_vars_path', help='Path to JSON file for ConfigurationManager')
    args = parser.parse_args()

    txaio.use_twisted()
    log = txaio.make_logger()
    txaio.start_logging()

    runner = ApplicationRunner(str(args.router), str(args.realm))
    runner.run(LocatorServiceComponent, auto_reconnect=True)
