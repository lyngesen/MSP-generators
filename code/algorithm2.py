''' script for answering research question 1:
RQ2: Given an MSP with |Y^1|, ..., |Y^S| what is the size of the minimum generator sequence Y_MIN?

For each problem MSP (from instances/problem)
    Find Yn and solve Minimum generator set sequence (MGSS) problem to find |Y_MIN|
    Save Yn and Y_MIN
'''

# local library imports
from classes import Point, PointList, MinkowskiSumProblem, KD_Node, KD_tree, MSPInstances
import methods
from timing import timeit, print_timeit, reset_timeit, time_object
from methods import N

from minimum_generator import solve_MGS_instance, SimpleFilter, build_model_covering, solve_model, display_solution, retrieve_solution_covering
# public library imports
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import os
import csv
import time
import itertools
import pprint

import sys
from multiprocessing import Pool, Process
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# timing

SimpleFilter = timeit(SimpleFilter)
time_object(KD_tree)
time_object(KD_Node)
time_object(PointList)
time_object(Point)
time_object(MinkowskiSumProblem,'MSP')
time_object(methods, prefix ="ALG")


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



def main_old():


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
    MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-200|200-mm-2_1.json')
    # MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-50|50|50-mmm-3_1.json')
    # MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-300|300|300-mmm-3_2.json')
    Y_list = MSP.Y_list
    if False:
        Y_list = MSP.Y_list
        Y_list[0] = Y_list[0]*2
        Y_list[1] = Y_list[1]*1.5
        # Y1 = PointList([(1,5), (2,4), (3,3), (4,2.5), (5,1)])
        Y1 = PointList([(1,5), (2,4), (3,3),(5,1)])
        Y2 = PointList([(8, 3),(9,2),(10,1)])
        Y_list = [Y1,Y2]
        MSP = MinkowskiSumProblem(Y_list)

    # MSP.plot()
    # plt.show()

    print(f"Running simple filter...")
    Yn, Yn_dict = SimpleFilter(Y_list)

    duplicates = {y_j : y_j_list for y_j, y_j_list in Yn_dict.items() if len(y_j_list)>1}
    duplicates = dict(sorted(duplicates.items(), key=lambda item: len(item[1]), reverse=True))

#     for y_j, y_j_list in Yn_dict.items():
        # if len(y_j_list) > 0:
            # print(f"{y_j=} ,  duplicates {len(y_j_list)}")
            # for y_j_I in y_j_list:
                # print(f"    {y_j_I}")

    
    Y_ms = methods.MS_sum(Y_list)
    print_timeit()
    

    print(f"{duplicates=}")

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
            print(f"{[y_j_I[s] in Y_generator_index[s] for s, _ in enumerate(Y_list)]}")
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
 



def alg2(MSP):
    ''' Algorithm 2 - Large IP model (SLOW) - For PhDSeminar CORAL May 2024 '''

    Y_MGS = solve_MGS_instance(MSP.Y_list, verbose = False, plot = False )

    return Y_MGS

@timeit
def get_fixed_and_reduced(C_dict, Y_list):

    # pprint.pprint(C_dict)

    Y_fixed = [set() for s, _ in enumerate(Y_list)]
    Y_reduced = [set() for s, _ in enumerate(Y_list)]

    for y, C in C_dict.items():
        for s,_ in enumerate(Y_list):
            is_fixed =True 
            ys = None

            for c in C:
                if ys == None:
                    ys = c[s]
                Y_reduced[s].add(c[s])
                if c[s] != ys:
                    is_fixed =False 
                    # break
            if is_fixed: # FINALLY
                Y_fixed[s].add(ys)
                

    if False:
        print(f"{[len(Y) for Y in Y_reduced]=}")
        print(f"{[len(Y) for Y in Y_fixed]=}")
        
        print(f"{Y_reduced=}")
        print(f"{Y_fixed=}")
        assert 1 == 0

    return Y_fixed, Y_reduced




def algorithm2(MSP):


    # print(f"Running simple filter...")
    Yn, C_dict = SimpleFilter(MSP.Y_list)

    # derive sets
    #   Y_fixed = nessesary points
    #   Y_reduced = subproblem points which contribute to at least one nondom solution
    Y_fixed, Y_reduced = get_fixed_and_reduced(C_dict, MSP.Y_list)

    # print(f"{[len(Y) for Y in Y_fixed]=}")
    # print(f"{[len(Y) for Y in Y_reduced]=}")
    # if the two sets are equal no further computation needed
    # print(f"{len(Yn)=}")
    # Y_fixed_pointlist = [[MSP.Y_list[s][i] for i,_ in enumerate(MSP.Y_list[s]) if i in Y_s_fixed] for s, Y_s_fixed in enumerate(MSP.Y_list)]
    
    @timeit
    def check_fixed_sufficient():
        Y_fixed_pointlist = [PointList([MSP.Y_list[s][i] for i in Y_fixed[s]]) for s in range(MSP.S)]
        Yn_fixed = methods.MS_sequential_filter(Y_fixed_pointlist)
        if Y_fixed == Y_reduced:
            return True, Yn_fixed
        if set(Yn_fixed.points).issubset(set(Yn.points)):
            return True, Yn_fixed
        else:
            return False, Yn_fixed

    check_val, Yn_fixed = check_fixed_sufficient()
    if check_val:
        Y_solution = Y_fixed
    else:
            # if the two sets are not equal a set partition problem must be solved
        if MSP.filename: # log that a covering problem as to be solved
            # with open('./instances/results/algorithm2_log', 'a') as logfile:
                # logfile.write('MSP.filename' + '\n')
            logger.info('covering problem solved: ' + MSP.filename)
        Yn_fixed_set = set(Yn_fixed.points)
        Yn_nongenerated = [y for y in Yn if y not in Yn_fixed_set]
        model = build_model_covering(MSP.Y_list,Yn, Yn_nongenerated, C_dict, Y_fixed, Y_reduced)
        solve_model(model)
        Y_chosen_dict = retrieve_solution_covering(model, MSP.Y_list)
        Y_solution = {s: Y_chosen_dict[s].union(set(Y_fixed[s])) for s in range(len(MSP.Y_list))}
        # print(f"{Y_chosen_dict=}")
        # print(f"{Y_solution=}")
    
    Y_MGS = [PointList([MSP.Y_list[s][i] for i in Y_solution[s]]) for s in range(MSP.S)]
    # print(f"{[len(Y) for Y in Y_MGS]=}")
    if True: # check of generating property
        Y_solution_pointlist = [PointList([MSP.Y_list[s][i] for i in Y_solution[s]]) for s in range(MSP.S)]
        Yn_solution = methods.MS_sequential_filter(Y_solution_pointlist)
        
        assert set(Yn_solution.points).issubset(set(Yn.points))


    return Y_MGS, len(Yn_solution)


# for logging
logname = 'algorithm2_log'
logging.basicConfig(level=logging.INFO, filename=logname)
logger = logging.getLogger(logname)


def algorithm2_run(MSP):


    time_start = time.time()
    # print(f"{MSP}")
    logger.info(MSP)
    Y_MGS, Yn_size = algorithm2(MSP)
    MGS_sizes = tuple([len(Y) for Y in Y_MGS])
    MGS_size = sum(MGS_sizes)
    str_out = f"{MSP.filename}, {MGS_size=},  rel_size={MGS_size/sum([len(Y) for Y in MSP.Y_list])*100:.0f}%,  {MGS_sizes=} \n \n"
    # print(str_out)
    logger.info(str_out)
    MGS = MinkowskiSumProblem(Y_MGS)

    MGS.filename = save_solution_dir + save_prefix + MSP.filename.split('/')[-1]


    # update run statistics
    statistics = {'filename':MSP.filename.split('/')[-1], 
                  'running_time': time.time() - time_start, 
                  'max_size': sum([len(Y) for Y in MSP.Y_list]),
                  'MGS_size': MGS_size, 
                  'MGS_sizes':MGS_sizes, 
                  'Yn_size':Yn_size, 
                  }

    MGS.statistics = statistics
    if 'timing' in sys.argv:
        print_timeit()
        reset_timeit()

    # print(f"{MGS.statistics=}")
    # print(f"{MGS.filename=}")


    MGS.save_json(MGS.filename)

    return statistics




save_prefix = 'alg2-'
save_solution_dir = './instances/results/algorithm2/'

def main():
    TestBank = MSPInstances('algorithm2', ignore_ifonly_l=True)
    # TestBank = MSPInstances(max_instances = 10, m_options = (4,), p_options = (4,))
    TestBank.filter_out_solved(save_prefix, save_solution_dir)

    print(f"{TestBank=}")

    time_start = time.time()    
    if 'multi' in sys.argv:
        with alive_bar(len(TestBank.filename_list), enrich_print=True) as bar, Pool() as pool:
            results = pool.imap_unordered(algorithm2_run, TestBank)
            for statistics in results:
                print(f"MSP:{statistics['filename']}, time: {statistics['running_time']}")
                bar()
    else:
        with alive_bar(len(TestBank.filename_list), enrich_print=False) as bar:
            for MSP in TestBank:
                statistics = algorithm2_run(MSP)
                bar()
                print(f"MSP:{statistics['filename']}, time: {statistics['running_time']}")

    print(f"total time: {time.time() - time_start}")

if __name__ == "__main__":
    main()
