.PHONY: test

test:
	py.test --junitxml=test-results.xml services/
