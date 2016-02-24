import os, sys
import pandas as pd
import numpy as np
import scipy as sp
from pandas import DataFrame, Series
import seaborn as sns
import pg8000
import time
import msgpack
import msgpack_numpy as m
m.patch()

from twisted.internet.defer import inlineCallbacks

from autobahn import wamp
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner


class Datacube(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        yield self.register(self)
        print("Ok, datacube procedures registered!")

    @wamp.register(u'org.alleninstitute.datacube.load')
    def load(self, arg=None):
        msg = msgpack.packb("test message", use_bin_type=True)
        print("msg", msg)
        return "loaded"


if __name__ == '__main__':

    import sys
    import argparse

    # load cell types data
    data = np.load('../data/ivscc.npy')

    # parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--web", type=int, default=8080,
                        help='Web port to use for embedded Web server. Use 0 to disable.')

    parser.add_argument("--router", type=str, default=None,
                        help='If given, connect to this WAMP router. Else run an embedded router on 9000.')

    args = parser.parse_args()

    from twisted.python import log
    log.startLogging(sys.stdout)

    # run WAMP application component
    from autobahn.twisted.wamp import ApplicationRunner
    runner = ApplicationRunner(url=u"ws://ibs-andys-ux3:8080/ws", realm=u"realm1")
    runner.run(Datacube)
