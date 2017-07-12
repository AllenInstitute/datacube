import sys
import os

from os import environ
from twisted.internet import threads
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from twisted.internet.defer import inlineCallbacks, returnValue

from configuration_manager import ConfigurationManager

from classes.surface_projection import SurfacePoint
from classes.projection_point import ProjectionPoint
from classes.filmstrip_locator import FilmStripLocator
from classes.model_loader import ModelLoader
from classes.voxel_lookup import VoxelLookup


class LocatorServiceComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        global ready
        
        try:
            path = os.path.join(os.path.dirname(__file__), "env_vars.json")
            config = ConfigurationManager(path)
        except (IOError) as e:
            print e.message

        


        ####################################################################
        ########################## Endpoints ###############################
        ####################################################################

        def describe ():
            result = 'Locator Service \n\n ------------------------ \n\n'
            result += 'get_surface_point([x,y,z]) => [x,y,z] \n' 

            return result


        def surface_point (seedPoint = None):
            results = dict()
            results.setdefault("success", False)

            # Validate Arguments
            if seedPoint == None:
                status = results.setdefault("status", dict())
                status.setdefault("message", "requires seed point")
                
                return results
            
            try:
                projector = SurfacePoint(config)
                results.setdefault("coordinate", projector.get(seedPoint))
                results['success'] = True

            except IOError as e:
                results.setdefault("message", e.message)

            return results

        def projection_point (path = None, pixel = None):
            results = dict()
            results.setdefault("success", False)

            # Validate Arguments
            if path == None or pixel == None or len(pixel) < 2:
                status = results.setdefault("status", dict())
                status.setdefault("message", "missing path or pixel argument(s)")

                return results

            try:
                projector = ProjectionPoint(config)
                results.setdefault("coordinate", projector.get(path, pixel))
                results["success"] = True

            except (IOError, ValueError) as e:
                results.setdefault("message", e.message)

            return results

        @inlineCallbacks
        def filmstrip_location (pixel = None, distanceMapPath = None, direction = None):
            results = dict()
            results.setdefault("success", False)

            # Validate Arguments
            if pixel == None or distanceMapPath == None or direction == None:
                status = results.setdefault("status", dict())
                status.setdefault("message", "missing pixel, distanceMapPath, or direction argument(s)")

            locator = FilmStripLocator(config)
            d = yield threads.deferToThread(locator.get, pixel, distanceMapPath, direction, results)
                
            returnValue(d)

        @inlineCallbacks
        def voxel_lookup (coords = None):
            results = dict()
            results.setdefault("success", False)
            
            lookup = VoxelLookup(config)
            
            d = yield threads.deferToThread(lookup.get, coords, results)

            returnValue(d)


        ready = False
        try:
            ####################################################################
            ###################### Endpoint Registration #######################
            ####################################################################

            yield self.register(describe,           u"org.brain_map.locator.describe")
            yield self.register(surface_point,      u"org.brain_map.locator.get_surface_point")
            yield self.register(projection_point,   u"org.brain_map.locator.get_projection_point")
            yield self.register(filmstrip_location, u"org.brain_map.locator.get_filmstrip_location")
            yield self.register(voxel_lookup,       u"org.brain_map.locator.get_voxel_structure")

            # TODO:
            # Streamlines
            # Reconstuctions

            ready = True
        except Exception as e:
            print("Could not register procedure: {0}".format(e))

        if ready:
            print ("Locator Ready!")
        else:
            print("Could not ready Locator Component")