import sys
import argparse
import json

from os import environ
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

from classes.surface_projection import SurfacePoint

class LocatorServiceComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        global ready

        def describe ():
            result = 'Locator Service \n\n ------------------------ \n\n'
            result += 'get_projection_point([x,y,z]) => [x,y,z] \n' 

            return result


        def get_projection_point (seedPoint = None):
            results = dict()

            # Return an error message if any arguments are missing
            if seedPoint is None:
                status = results.setdefault("status", dict())
                results.setdefault("success", False)
                status.setdefault("message", "requires seed point")
                
                return json.dumps(results)
            
            # If all arguments are present instantiate the approp. object and get the infos
            projector = SurfacePoint()
            return projector.get(seedPoint)

        ready = False
        try:
            yield self.register(describe, u"org.alleninstitute.locator.describe")
            yield self.register(get_projection_point, u"org.alleninstitute.locator.get_projection_point")
            ready = True
        except Exception as e:
            print("Could not register procedure: {0}".format(e))

        if ready:
            print ("Locator Ready!")
        else:
            print("Could not ready Locator Component")