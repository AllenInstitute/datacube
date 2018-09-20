from __future__ import absolute_import

#todo: not the right way to do this
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from twisted.trial import unittest

from mock import patch, Mock

from autobahn.twisted.wamp import ApplicationRunner

import server

class TestDatacubeServiceComponent(unittest.TestCase):
    @patch('twisted.internet.reactor')
    def test_test(self, fakereactor):
        c = server.DatacubeServiceComponent()
        self.assertEqual(1, c.test())
