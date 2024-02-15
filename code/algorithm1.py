''' script for answering research question 1:
RQ1: How large is |Yn| given |Yn^s| for s in S

For each problem MSP (from instances/problem)
    Find Yn and save solution (PointList) to json file and add statistics with |Yn|
'''

from classes import Point, PointList, MinkowskiSumProblem, KD_Node, KD_tree
import methods
from methods import N
import matplotlib.pyplot as plt
# import pyomo.environ as pyomo
from timing import timeit, print_timeit, reset_timeit
import os
import csv


methods.MS_doubling_filter = timeit(methods.MS_doubling_filter)
methods.MS_sequential_filter = timeit(methods.MS_sequential_filter)
methods.MS_naive_filter = timeit(methods.MS_naive_filter)
methods.MS_sum = timeit(methods.MS_sum)
methods.KD_filter = timeit(methods.KD_filter)


methods.lex_sort = timeit(methods.lex_sort)
methods.two_phase_filter = timeit(methods.two_phase_filter)
methods.N = timeit(methods.N, "filter N")



Point.__le__ = timeit(Point.__le__)
Point.__lt__ = timeit(Point.__lt__)
Point.__gt__ = timeit(Point.__gt__)

PointList.__add__ = timeit(PointList.__add__)
methods.MS_doubling_filter = timeit(methods.MS_doubling_filter)



PointList.save_json = timeit(PointList.save_json)
PointList.from_json = timeit(PointList.from_json, 'PointList.from_json')
MinkowskiSumProblem.from_json = timeit(MinkowskiSumProblem.from_json, 'MSP.from_json')


def main():


    # problem_name = "prob-3-50|50-ul-2_1.json"
    all_problems = os.listdir("instances/problems/")

    def get_Y_max_size(MSP: MinkowskiSumProblem):
        problem_count = MSP.filename.count('|')
        problem_sizes = MSP.filename.split('|')
        problem_sizes[0] = problem_sizes[0].split("-")[-1]
        problem_sizes[-1] = problem_sizes[-1].split("-")[0]
        problem_sizes = [int(problem_size) for problem_size in problem_sizes]
        total_size = 1
        for problem_size in problem_sizes:
            total_size *= problem_size
        return total_size

    # all_problems = [problem_name for problem_name in all_problems if "50" in problem_name]
    # all_problems = [problem_name for problem_name in all_problems if " in problem_name]
    # print(f"{all_problems[:10]=}")
    for problem_name in all_problems:
        if problem_name in os.listdir("instances/results/algorithm1/"):
            print(f"  problem solved - skipping {problem_name}")
            continue
        MSP = MinkowskiSumProblem.from_json("instances/problems/"+ problem_name) 
        print(f"{MSP=}")
        print(f"{get_Y_max_size(MSP)=}")

        if MSP.dim != 2:
            continue

        if get_Y_max_size(MSP) > 1_000_000:
            print(f"  problem too big (skipping): {get_Y_max_size(MSP)=}")
            continue

        Yn = methods.MS_sequential_filter(MSP.Y_list)
        # Yn = methods.MS_sequential_filter(MSP.Y_list)
        Yn.save_json("instances/results/algorithm1/alg1-" + problem_name)
        print(f"{len(Yn)=}")

        print_timeit()
        reset_timeit()
        print(" ")

def test_times():

    MSP = MinkowskiSumProblem.from_json("instances/problems/prob-3-200|200-uu-2_1.json") 
    MSP = MinkowskiSumProblem.from_json("instances/problems/prob-3-200|200-uu-2_1.json") 
    # MSP = MinkowskiSumProblem.from_json("instances/problems/prob-3-50|50-ll-2_1.json") 

      
    print(f"{MSP=}")
    Yn = methods.MS_sequential_filter(MSP.Y_list, filter_alg = methods.KD_filter)
    print_timeit()
    print(f"{len(Yn)=}")
    reset_timeit()
    Yn = methods.MS_sequential_filter(MSP.Y_list, filter_alg = methods.naive_filter)
    print_timeit()

    print(f"{len(Yn)=}")

def json_files_to_csv():

    with open('instances/results/RQ1.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter = ";")
        


        # writer.writerow(";".join(("problem", "dim", "|Yn|")))
        writer.writerow(("problem", "dim", "|Yn|"))


        for json_file in os.listdir("instances/results/algorithm1/"):
            Yn = PointList.from_json( "instances/results/algorithm1/" + json_file)
            row_data = [json_file, str(Yn.dim), str(len(Yn)) ]
            # row = ";".join(row_data)
            writer.writerow(row_data)





if __name__ == "__main__":
    # test_times()
    # example_Yn()
    main()
    json_files_to_csv()
