import sys
import argparse
import json

from os import environ
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

from classes.surface_projection import SurfacePoint
from classes.projection_point import ProjectionPoint

class LocatorServiceComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        global ready

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

            # Return an error message if any arguments are missing
            if seedPoint is None:
                status = results.setdefault("status", dict())
                status.setdefault("message", "requires seed point")
                
                return json.dumps(results)
            
            try:
                # If all arguments are present instantiate the approp. object and get the infos
                projector = SurfacePoint()
                results.setdefault("coordinate", projector.get(seedPoint))
                results['success'] = True
                
            except Exception as e:
                results.setdefault("message", e.message)

            return json.dumps(results)

        def projection_point (path = None, pixel = None):
            results = dict()
            results.setdefault("success", False)

            if path is None or pixel is None or len(pixel) < 2:
                status = results.setdefault("status", dict())
                status.setdefault("message", "missing path or pixel arguments")

                return json.dumps(results)

            try:
                projector = ProjectionPoint()
                results.setdefault("coordinate", projector.get(path, pixel))
                results["success"] = True

            except (FileNotFoundError, ValueError) as e:
                results.setdefault("message", e.message)

           
            return json.dumps(results)


        ready = False
        try:
            ####################################################################
            ###################### Endpoint Registration #######################
            ####################################################################

            yield self.register(describe, u"org.alleninstitute.locator.describe")
            yield self.register(surface_point, u"org.alleninstitute.locator.get_surface_point")
            yield self.register(projection_point, u"org.alleninstitute.locator.get_projection_point")

            ready = True
        except Exception as e:
            print("Could not register procedure: {0}".format(e))

        if ready:
            print ("Locator Ready!")
        else:
            print("Could not ready Locator Component")