from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
import itertools
import random
import time
import argparse

class MyComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        res = yield self.call('org.alleninstitute.pandas_service.filter_cell_specimens', fields='indexes_only')
        max_idx = res['filtered_total']
        for _ in itertools.repeat(None, args.num_requests):
            indexes = random.sample(range(max_idx), args.page_size)
            start = time.time()
            res = yield self.call('org.alleninstitute.pandas_service.filter_cell_specimens', indexes=indexes)
            print(str((time.time() - start) * 1000.0) + ' ms')
        from twisted.internet import reactor
        reactor.stop()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--page-size', default=100, type=int)
    parser.add_argument('--num-requests', default=100, type=int)
    args = parser.parse_args()

    runner = ApplicationRunner(url=u"ws://ibs-chrisba-ux1:8081/ws", realm=u"aibs")
    runner.run(MyComponent)
