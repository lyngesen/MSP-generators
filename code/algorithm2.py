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

from minimum_generator import solve_instance, SimpleFilter
# public library imports
import matplotlib.pyplot as plt
import os
import csv
import time


def plot_y_j(y_j, Y_list, marker = 'x'):
    y_j.plot(marker='x', color='black')
    for s, Y in enumerate(Y_list):
        Y[y_j.index_list[s]].plot(marker=marker, color='black')
    y_ms = Point([0 for _ in range(Y_list[0].dim)])
    Y_j = [y_ms]
    for s,Y in enumerate(Y_list):
        y_ms = y_ms + Y[y_j.index_list[s]]
        Y_j.append(y_ms)
    Y_j = PointList(Y_j)
    Y_j.plot(l='Y_j', line=True)



def main():


    # MSP = MinkowskiSumProblem.from_json('instances/problems/prob-3-100|100|100-mmm-3_1.json')
    MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-50|50-mm-2_1.json')
    print(f"{MSP}")
    Y_list = MSP
    Y_list = []
    Y_list.append(PointList.from_json('instances/subproblems/sp-2-10-u_1.json'))
    Y_list.append(PointList.from_json('instances/subproblems/sp-2-10-l_2.json'))
    Y_list.append(PointList.from_json('instances/subproblems/sp-2-10-m_2.json'))
    

    # Y_MIN_LIST = solve_instance(MSP.Y_list, plot= False)

    MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-200|200-ul-2_1.json')
    # MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-50|50|50-mmm-3_1.json')
    # MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-300|300|300-mmm-3_2.json')
    Y_list = MSP.Y_list
    if True:
        Y_list = MSP.Y_list
        Y_list[0] = Y_list[0]*2
        Y_list[1] = Y_list[1]*1.5
        # Y1 = PointList([(1,5), (2,4), (3,3), (4,2.5), (5,1)])
        Y1 = PointList([(1,5), (2,4), (3,3),(5,1)])
        Y2 = PointList([(8, 3),(9,2),(10,1)])
        Y_list = [Y1,Y2]
        MSP = MinkowskiSumProblem(Y_list)

    print(f"Running simple filter...")
    Yn, Y_ms, Yn_dict = SimpleFilter(Y_list)

    duplicates = {y_j : y_j_list for y_j, y_j_list in Yn_dict.items() if len(y_j_list)>1}
    duplicates = dict(sorted(duplicates.items(), key=lambda item: len(item[1]), reverse=True))

#     for y_j, y_j_list in Yn_dict.items():
        # if len(y_j_list) > 0:
            # print(f"{y_j=} ,  duplicates {len(y_j_list)}")
            # for y_j_I in y_j_list:
                # print(f"    {y_j_I}")

    
        

    print(f"{len(Y_ms)=}")
    print(f"{len(Yn)=}")
    print(f"{len(duplicates.keys())=}")

    print_timeit()
    


    Y_generator_index = {s:set() for s, _ in enumerate(Y_list)}
    for y_j, y_j_list in Yn_dict.items():
        if y_j in duplicates.keys(): continue
        assert len( y_j_list) == 1
        y_j_I =  y_j_list[0]
        for s, Ys in enumerate(Y_list):
            Y_generator_index[s].add(y_j_I[s])
            # if y_j_I not in Y_gen
        
    # print(f"{Y_generator_index=}")
    Yn_generated = {y_j for y_j in Yn_dict if y_j not in duplicates}
    print(f"Generated points of Yn {len(Yn_generated)} of {len(Yn)}")
    for y_j, y_j_list in duplicates.items():
        for y_j_I in y_j_list:
            # print(f"{[y_j_I[s] in Y_generator_index[s] for s, _ in enumerate(Y_list)]}")
            if all([y_j_I[s] in Y_generator_index[s] for s, _ in enumerate(Y_list)]):
                Yn_generated.add(y_j)
                break

    print(f"Generated points of Yn {len(Yn_generated)} of {len(Yn)}")
    for s, Y in enumerate(Y_list):
        print(f"|Y{s+1}| = {len(Y)}")
        print(f"|Yg{s+1}| = {len(Y_generator_index[s])}")

    # print(f"{list(duplicates.items())[0]}")
    # for y_j in list(Y_ms)[::len(Yn)//3]:


    for y_j, y_j_list in list(duplicates.items())[:3:]:
    # for y_j, y_j_list in Yn_dict.items():
        # methods.MS_sum(MSP.Y_list).plot()
        MSP.plot()
        Yn.plot(f"$Yn$")
        for k, y_j_I in enumerate(y_j_list):
            y_j.index_list = y_j_I
            plot_y_j(y_j, Y_list, marker=k)

        plt.show()
    # save solution
    


if __name__ == "__main__":
    main()
