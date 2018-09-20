from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
import itertools
import random
import time
import json
import sys
from twisted.internet import reactor
from random import randint
from wamp import ApplicationSession, ApplicationRunner

class MyComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        offset = randint(0,0)
        with open(sys.argv[1]) as f:
            for line in f:
                time_s = line[:5]
                minutes = time_s[:2]
                seconds = time_s[3:]
                elapsed = int(minutes)*60 + int(seconds) - offset

                request_str = line[6:]
                request = json.loads(request_str)

                @inlineCallbacks
                def time_request(request):
                    start = time.time()
                    if True: #request.get('sort', []) or request.get('filters', []):
                        res = yield self.call(u'org.alleninstitute.pandas_service.filter_cell_specimens', **request)
                    else:
                        res = yield self.call(u'org.alleninstitute.pandas_service.get_cell_specimens', **request)
                    #print(request, res)
                    print(time.time() - start)

                #reactor.callLater(elapsed, self.call, 'org.alleninstitute.pandas_service.filter_cell_specimens', **request)
                reactor.callLater(elapsed, time_request, request)
        reactor.callLater(elapsed + 10, reactor.stop)
        yield None
            

if __name__ == "__main__":
    runner = ApplicationRunner(url=unicode(sys.argv[2]), realm=u"aibs")
    runner.run(MyComponent)
