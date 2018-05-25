CONN_BRIDGE_PORT ?= 39608
export CONN_BRIDGE_PORT

DATACUBE_WAMP_TRANSPORT ?= "ws://devdatacube:8080/ws"
export DATACUBE_WAMP_TRANSPORT

DATACUBE_WAMP_REALM ?= "aibs"
export DATACUBE_WAMP_REALM

test: pytest_unit pytest_integration

pytest_unit:
	pipenv install --dev || exit 0
	pipenv run python -m pytest test/unit --cov=legacy_routes/ --cov-report=html --cov-config=coveragerc

pytest_integration:
	pipenv install --dev || exit 0
	pipenv run python conn_bridge.py --port $(CONN_BRIDGE_PORT) --wamp_transport $(DATACUBE_WAMP_TRANSPORT) --wamp_realm $(DATACUBE_WAMP_REALM) &
	pipenv run python -m pytest test/integration --tb=short  --ignore test/integration/cache_integration_test_data.py || exit 0
	kill $$(ps aux | grep "[c]onn_bridge.py --port $(CONN_BRIDGE_PORT)" | awk '{print $$2}')
