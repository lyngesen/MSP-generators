''' script for answering research question 1:
RQ2: Given an MSP with |Y^1|, ..., |Y^S| what is the size of the minimum generator sequence Y_MIN?

For each problem MSP (from instances/problem)
    Find Yn and solve Minimum generator set sequence (MGSS) problem to find |Y_MIN|
    Save Yn and Y_MIN
'''

# local library imports
from classes import Point, PointList, MinkowskiSumProblem, KD_Node, KD_tree
import methods
from timing import timeit, print_timeit, reset_timeit
from methods import N

from minimum_generator import solve_instance
# public library imports
import matplotlib.pyplot as plt
import os
import csv
import time



def main():


    MSP = MinkowskiSumProblem.from_json('instances/problems/prob-3-300|300|300|300|300-lllll-5_4.json')
    print(f"{MSP}")
    Y_MIN_LIST = solve_instance(MSP.Y_list, plot= False)

    
    print_timeit()
    
    # save solution
    


if __name__ == "__main__":
    main()
