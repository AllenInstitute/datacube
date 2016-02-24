Datacube Service
==========================

WAMP component providing a datacube service.

A browser based UI is included which uses AutobahnJS to access
the datacube service.

Further, since it's a standard WAMP service, any WAMP client can use
the service. You could access it i.e. from a native
Android app via AutobahnAndroid or from a remote AutobahnPython based
client.


Installation
-------

TODO

Running
-------

Run the datacube backend component by doing

    python server.py

and open

    http://localhost:8080/

in your browser.

To activate debug output, start it

    python server.py --debug



Background
----------

The datacube service performs datacube statistical queries.

