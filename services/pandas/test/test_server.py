from __future__ import absolute_import

#todo: not the right way to do this
import sys
sys.path.insert(0, '.')

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from twisted.trial import unittest

from mock import patch, Mock

from autobahn.twisted.wamp import ApplicationRunner

import server

class TestPandasServiceComponent(unittest.TestCase):
    @patch('twisted.internet.reactor')
    def test_test(self, fakereactor):
        c = server.PandasServiceComponent()
        self.assertEqual(1, c.test())
