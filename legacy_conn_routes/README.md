# legacy_conn_routes
Exposes a mouseconn-like http interface to the datacube. Specifically supports:
- search/injection_rows


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
