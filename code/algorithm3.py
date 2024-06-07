"""
File containing code for algorithm3 answering empirical research question 3

Usage:
    Run to add computational results
    output saved in: ./instances/results/algorithm3/result.csv
"""

from classes import Point, PointList, LinkedList, MinkowskiSumProblem
import methods
from methods import N
import time
import csv
import math
import os
from minimum_generator import solve_instance

def induced_UB_plot(level, Y1,Y2, prefix='', plot=True):
    print(f"{prefix}")
    # print(f"{level=}")
    def get_partial(Y, level='all'):   
        Y = N(Y)
        Y2e_points = [y for y in Y if y.cls == 'se']
        Y2other_points = [y for y in Y if y.cls != 'se']
        # random.shuffle(Y2other_points)
        match level:
            case 'all':
                return Y
            case 'lexmin': 
                return PointList((Y[0], Y[-1]))
            case 'extreme':
                return PointList(Y2e_points)
            # case float():
            case _:
                to_index = math.floor(float(level)*len(Y2other_points))
                return PointList(Y2e_points + Y2other_points[:to_index])
                # print(f"case not implemented {level}")
    
    
    Y2_partial = get_partial(Y2, level)


    ub_time = time.time()
    U = methods.find_generator_U(Y2_partial, Y1)
    ub_time = time.time() - ub_time

    Uline = methods.induced_UB(U,line=True)
   
    Y2_dominated = [y for y in Y2 if y.cls != 'se' and U.dominates_point(y)]
    dominated_relative = len(Y2_dominated)/len(Y2)
    print(f"dominated: {len(Y2_dominated)} \nrelative: {dominated_relative*100}\%")
    
    run_data = {'prefix' : prefix,
                'Y1_size' : len(Y1),
                'Y2_size' : len(Y2),
                'U' : len(U),
                'U_time' : ub_time,
                'dominated_points' : len(Y2_dominated),
                'dominated_relative_Y2' : dominated_relative,
                }

    return run_data
def multiple_induced_UB():


    set_options = ['l','m','u']
    size_options = [10, 50, 100, 150, 200, 300, 600]
    seed_options = [1,2,3,4,5]
    # UB_options = ['lexmin','extreme','0.25','0.5','0.75','all']
    UB_options = ['extreme']

    csv_file_path = './instances/results/algorithm3/result_slides_alg1_2.csv'
    # get last row
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            lastrow = row
    start_runs = False

    for s1 in size_options:
        s2 = s1
        for ub_level in UB_options:
        # s1 = 100 
        # s2 = 100
            # for s2 in size_options:
            for t1 in set_options:
                for t2 in set_options:
                    for seed in seed_options:
                        prefix = f'{t1}-{t2}_{s1}_{s2}_{ub_level}_{seed}_'
#                         if start_runs == False:
                            # if prefix == lastrow[0]:
                                # start_runs = True
                                # print(f"Starting run after {prefix}")
                            # continue
                        Y1 = PointList.from_json(f"./instances/subproblems/sp-2-{s1}-{t1}_{seed}.json")
                        Y2 = PointList.from_json(f"./instances/subproblems/sp-2-{s2}-{t2}_{max(seed_options)+1-seed}.json")
                        data = induced_UB_plot(ub_level, Y1,Y2, prefix, plot=False) 
                        data.update({'t1':t1, 't2':t2, 's1':s1, 's2':s2,'seed':seed,'ub_level':ub_level})
                        print(f"ALG1 solving Yn")
                        Y = Y1+Y2
                        Yn = N(Y)
                        data.update({'Y_size':len(Y), 'Yn_size':len(Yn)})
                        print(f"Solving MSG")
                        G = solve_instance([Y1,Y2])
                        data.update({'G1_size':len(G[0]), 'G2_size':len(G[1])})
                        with open(csv_file_path, 'a') as csv_file:
                            # add header if file empty
                            writer = csv.writer(csv_file)
                            if os.path.getsize(csv_file_path) == 0:
                                writer.writerow(data.keys())
                            writer.writerow(data.values())

def main():

    multiple_induced_UB()

if __name__ == '__main__':
    
    main()
