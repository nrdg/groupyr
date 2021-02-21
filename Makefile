.PHONY: clean clean-test clean-pyc clean-build lint help
.DEFAULT_GOAL := help

lint:         ## Check code style
	flake8
	black --check .
	pydocstyle

test:         ## Run unit tests using pytest
	pytest --pyargs groupyr --cov-report term-missing --cov-config .coveragerc --cov=groupyr -n auto

devtest:      ## Run unit tests using pytest and abort testing after first failure
    # Unit testing with the -x option, aborts testing after first failure
    # Useful for development when tests are long
	pytest -x --pyargs groupyr --cov-report term-missing --cov-config .coveragerc --cov=groupyr -n auto

clean:        ## Remove all build, test, coverage and Python artifacts
clean: clean-build clean-pyc

clean-build:  ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:    ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

release:      ## Package and upload a release
release: dist
	twine upload dist/*

dist:         ## Build source and wheel package
dist: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	ls -l dist

help:         ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
