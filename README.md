# Generator sets for Minkowski Sum Problems

This repository is the official code implementation of [Generator Sets for the Minkowski Sum Problem - Theory and Insights](https://arxiv.org/). 


# Structure 

- code
  - instances
  - algorithm1.py
  - algorithm3.py
- docs
  - index.html
  - python_coverage
  - python_documentation
  - algorithm1.html
  - algorithm2.html
  - algorithm3.html
  - problem-generation.html
  - problem-statistics.html


## Documentation

Documentation of the python modules are located in [docs/python_documentation](./docs/python_documentation).

Unittest and coverage reports are located in [docs/python_coverage](./docs/)

## Results

A summary of the results used in the paper can be found in the directory `docs`:
[LINK](./docs/)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To use the implementation of [Algorithm1] for finding Minimum Generator Sets some instances require a MIP solver. The default solver used is specified in the file `code/minimum_generator.py`. Change this line to any solver installed on your system compatible with `Pyomo` (Most solvers are compatible).

The program will use a Python non-dominance filter. The non-dominance filter used in the paper utilized a c-implementation provided by Bruno Lang. When this is made public a link to the repository will be provided here.


## Usage

The code provide several scripts which can be run as main.

Python scripts:

- `python3 plots.py SHOW` - run the main in plots, by default this shows an interactive Minkowski Sum Problem interface.
- `python3 algorithm1.py `

```
usage: algorithm1.py [-h] [-timelimit TIMELIMIT] [-npartition NPARTITION] [-kpartition KPARTITION] [-memorylimit MEMORYLIMIT] [-outdir OUTDIR]
                     [-logpath LOGPATH] [-msppreset MSPPRESET] [-solveall] [-alg2] [-skipYn]

Save instance results PointList in dir.

options:
  -h, --help            show this help message and exit
  -timelimit TIMELIMIT  Time limit for each instance
  -npartition NPARTITION
                        Total partitions (n) of test instances
  -kpartition KPARTITION
                        Number of specific test intsance partition
  -memorylimit MEMORYLIMIT
                        Memory limit for each instance
  -outdir OUTDIR        Result dir, where instances are saved
  -logpath LOGPATH      path where log (algorithm1.log) files are to be saved
  -msppreset MSPPRESET  Choice of preset instances to solve default: algorithm1. other choices grendel_test, algorithm2
  -solveall             if flag added, all instances are solved (already solved instances will not be filtered out)
  -alg2                 if flag added, MGS will be solved using algorithm2)
  -skipYn               if flag added, the found Yn sets will be saved

```




## Test Instances

All test instances are located in `./code/instances/`. See [MO-repo-Lyngesen24](https://github.com/MCDMSociety/MOrepo-Lyngesen24) for descriptions of the instances.
