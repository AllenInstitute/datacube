from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
import itertools
import random
import time
import argparse

class MyComponent(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        import json
        filters = json.loads('''[
            {
                "field": "p_sg",
                "op": "<=",
                "value": 0.01
            },
            {
                "field": "pref_phase_sg",
                "op": "=",
                "value": 0.25
            },
            {
                "field": "pref_sf_sg",
                "op": "=",
                "value": 0.02
            },
            {
                "field": "p_dg",
                "op": "<=",
                "value": 0.01
            },
            {
                "field": "pref_tf_dg",
                "op": "=",
                "value": 1
            },
            {
                "field": "dsi_dg",
                "op": "between",
                "value": [
                    0.7,
                    2
                ]
            },
            {
                "field": "time_to_peak_ns",
                "op": "between",
                "value": [
                    0,
                    0.18
                ]
            },
            {
                "field": "area",
                "op": "in",
                "value": [
                    "VISp",
                    "VISal",
                    "VISl"
                ]
            }
        ]''')

        res = yield self.call('org.alleninstitute.pandas_service.filter_cell_specimens', fields='indexes_only')
        max_idx = res['filtered_total']
        for repeat in range(args.num_requests):
            start = time.time()
            if repeat % 10 == 0:
                res = yield self.call('org.alleninstitute.pandas_service.filter_cell_specimens', filters=filters, fields='indexes_only')
            else:
                indexes = random.sample(range(max_idx), args.page_size)
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
