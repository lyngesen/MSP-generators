# Tests

Run all tests from project root `code/` using Pytest by running the command.
* `python -m pytest --cov`
* `python -m pytest --cov-report html --cov=..`
* `pytest --cov-report html --cov=..`

* Run in virtual enviroment.
* `python -m pytest` run pytest as python module. 
  * Requires `pip install pytest`
* `--cov=..` use the plugin [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/readme.html#coverage-data-file) to generate coverage report of files in folder `..`(parent).
* `--cov-report html`, save report as html.
  * Requires `pip install pytest-cov`

Continously add new tests when functions are added.
