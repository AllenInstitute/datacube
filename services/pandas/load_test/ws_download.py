from os import environ
from twisted.internet import defer, reactor, task
from twisted.internet.defer import inlineCallbacks

from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.types import CallOptions

import time

class Component(ApplicationSession):

    def get_records(self, out, start_time):
        numrecords = 45287
        pagesize = 500
        for i in range(0, numrecords, pagesize):
            start = i
            stop = min(i+(pagesize-1), numrecords)
            d = self.call(u'org.brain-map.api.datacube.raw.cell_specimens', select={'dim_0': {'start': start, 'stop': stop}})
            out.append(None)
            d.addCallback(self.process_records, out, len(out)-1, start_time)
            yield d

    def process_records(self, res, out, i, start_time):
        print('[' + ''.join([' ' if x is None else 'x' for x in out]) + ']')
        print(time.time()-start_time)
        out[i] = res

    @inlineCallbacks
    def onJoin(self, details):

        try:
            import time
            start_time = time.time()

            out = []
            deferreds = []
            concurrent = 8
            coop = task.Cooperator()
            work = self.get_records(out, start_time)
            for i in range(concurrent):
                d = coop.coiterate(work)
                deferreds.append(d)
            d1 = defer.DeferredList(deferreds)
            res = yield d1
            print('[' + ''.join([' ' if x is None else 'x' for x in out]) + ']')
            print('total: ' + str(time.time()-start_time))

        except Exception as e:
            print(str(e))
        finally:
            self.leave()

    def onDisconnect(self):
        print("disconnected")
        reactor.stop()


if __name__ == '__main__':
    runner = ApplicationRunner(u"ws://devdatacube:8080/ws", u"aibs")
    runner.run(Component)
