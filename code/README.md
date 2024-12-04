# Code used for the Minkowski sum article

* `classes.py` - Implementation of MO classes: `Point`, `PointList`, `MinkowskiSumProblem` and `TestInstances`.
* `generator.py` - Generator of testsets saved in directory `testsets/`.
* `methods.py` - Implementation of sorting and filtering algorithms.
* `algorithm1.py` - Main for running empirical study (filtering algorithm) for all MSPs.
* `algorithm1.py -alg2` - Main for running filtering algorithm along with the MGS (Algorithm 1 in paper) for all MSPs.
* `algorithm3.py` - Main for running script for finding reduced generator sets (Algorithm 2 in paper) for all MSPs with two objectives.
* `plots.py` - Main script for generating plots.
* `timing.py` - Timing decorator.
* `requirements.txt` - Third party libraries used in the project. Generated using `pipreqs .`.

# Documentation

Documentation of python implementation can be found in [Docs](https://lyngesen.github.io/MSP-generators/index.html).

# Unittests and coverage report

* `./tests/` - contains unittests.

Coverage reports are located in [Docs](https://lyngesen.github.io/MSP-generators/index.html).

