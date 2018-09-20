.PHONY: test

test:
	py.test --ignore=services/legacy_conn_routes/test/integration --junitxml=test-results.xml services/
