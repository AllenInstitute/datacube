# legacy_conn_routes
Exposes a mouseconn-like http interface to the datacube.

To start the bridge, use `make run` from this directory or `crossbar start` from the parent directory.

### Old information on starting the bridge
The bridge now starts via crossbar start. If for some reason you want to start it standalone, you can do so as 
described below (or check the makefile).

To run the tests:
```
make test
```
To recache the test data:
```
python test/integration/cache_integration_test_data.py
```
To start the server:
```
python conn_bridge.py
```
