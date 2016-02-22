from os import environ
import sys
import decimal

from twisted.internet.defer import inlineCallbacks

from autobahn import wamp
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner


class Datacube(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):
        self.clear()
        yield self.register(self)
        print("Ok, datacube procedures registered!")

    @wamp.register(u'org.alleninstitute.datacube.clear')
    def clear(self, arg=None):
        self.op = None
        self.current = decimal.Decimal(0)
        return str(self.current)

    @wamp.register(u'org.alleninstitute.datacube.calc')
    def calc(self, op, num):
        num = decimal.Decimal(num)
        if self.op:
            if self.op == "+":
                self.current += num
            elif self.op == "-":
                self.current -= num
            elif self.op == "*":
                self.current *= num
            elif self.op == "/":
                self.current /= num
            self.op = op
        else:
            self.op = op
            self.current = num

        res = str(self.current)
        if op == "=":
            self.clear()

        return res


if __name__ == '__main__':

    decimal.getcontext().prec = 20

    import sys
    import argparse

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
