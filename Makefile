all:

test:
	mkdir -p test-results
	python3 -m pytest \
	    --cov=hpnevergrad \
	    --no-cov-on-fail \
	    --cov-report=html:test-results/htmlcov \
	    --cov-report term \
	    --doctest-modules \
	    --junitxml=test-results/junit.xml \
	    hpnevergrad tests
	python3 -m coverage xml -o test-results/coverage.xml


format:
	autoflake -r -i examples hpnevergrad tests
	isort -rc examples hpnevergrad tests
	yapf -i --style pep8 --recursive examples hpnevergrad tests

style-check:
	mypy --ignore-missing-imports hpnevergrad
	flake8 --ignore E501,E203,F401,W503,W504,F841 hpnevergrad examples tests

serve-coverage-report:
	cd test-results/htmlcov && python3 -m http.server

wheel:
	python3 setup.py sdist bdist_wheel

doc:
	cd docs && ./gendoc.sh

install:
	# install prerequisites
	# TODO:
	#   1. install requirments
	#   2. install pre-commit hook
