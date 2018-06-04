# legacy_conn_routes
Exposes a mouseconn-like http interface to the datacube.

Please note that the Makefile specifies `PIPENV_VENV_IN_PROJECT=1`, causing pipenv to produce a virtualenv local to this 
project folder. This might fail when running the bridge from a conda environement. In order to get around this issue you can 
disable the local virtualenv by exporting `PIPENV_VENV_IN_PROJECT=0` or by installing virtualenv using conda.

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
