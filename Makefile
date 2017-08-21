.PHONY: test

test:
	py.test --junitxml=test-reports/output.xml
