''' script for answering research question 1:
RQ1: How large is |Yn| given |Yn^s| for s in S

For each problem MSP (from instances/problem)
    Find Yn and save solution (PointList) to json file and add statistics with |Yn|
'''

from classes import Point, PointList, MinkowskiSumProblem, KD_Node, KD_tree
import methods
import timing
from timing import timeit, print_timeit, reset_timeit, time_object
from methods import N
# public library imports
import matplotlib.pyplot as plt
import os
import csv
import time

time_object(KD_tree)
time_object(KD_Node)
time_object(PointList)
time_object(Point)
time_object(timing)
time_object(MinkowskiSumProblem,'MSP')
time_object(methods, prefix ="ALG")


def name_dict(problem_file):
    filename = problem_file
    problem_file = problem_file.split(".json")[0]
    problem_file, seed = problem_file.split("_")
    _, p, size, method, M = problem_file.split("-")
    size = size.split("|")[0]
    p, M, size = int(p), int(M), int(size)
    D = {'filename': filename, 'p': p, 'method':method, 'M': M, 'size': size}
    return D
    
def name_dict_keys(problem_file):
    D = name_dict(problem_file)
    return (D['size'], D['p'], D['M'])

def sorted_problems():
    all_problems = os.listdir("instances/problems/")

    print(f"{name_dict(all_problems[0])=}")

    all_problems = sorted(all_problems, key = name_dict_keys )

    for p in all_problems[:100]:
        # print(f"{name_dict(p)}")
        print(" ".join(f"{k} = {v},\t" for k, v in name_dict(p).items()))
    # print(f"{all_problems[:10]=}")



def main():


    # problem_name = "prob-3-50|50-ul-2_1.json"
    all_problems = os.listdir("instances/problems/")
    all_problems = sorted(all_problems, key = name_dict_keys )


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
    save_prefix = "alg1-"
    for problem_name in all_problems:
        print(" ".join(f"{k} = {v},\t" for k, v in name_dict(problem_name).items()))

        if problem_name.split(".")[-1] != "json": continue
        if save_prefix + problem_name in os.listdir("instances/results/algorithm1/"):
            print(f"  problem solved - skipping {problem_name}")
            continue
        MSP = MinkowskiSumProblem.from_json("instances/problems/"+ problem_name) 

        # if MSP.dim == 2:
            # continue

        if get_Y_max_size(MSP) > 100_000_000:
            print(f"  problem too big (skipping): {get_Y_max_size(MSP)=}")
            continue

        # print(f"{MSP=}")
        print(f"{get_Y_max_size(MSP)=}")
        filter_time = time.time()
        Yn = methods.MS_sequential_filter(MSP.Y_list)
        Yn.statistics['filter_time'] = time.time() - filter_time
        # Yn = methods.MS_sequential_filter(MSP.Y_list)
        Yn.save_json("instances/results/algorithm1/"  +  save_prefix + problem_name)
        print(f"{len(Yn)=}")

        print_timeit()
        reset_timeit()
        print(" ")

def test_times():

    # MSP = MinkowskiSumProblem.from_json("instances/problems/prob-2-50|50-ll-2_1.json") 
    # MSP = MinkowskiSumProblem.from_json("instances/problems/prob-5-300|300|300|300|300-lllll-5_3.json") 
    MSP = MinkowskiSumProblem.from_json("instances/problems/prob-3-100|100|100-mmm-3_1.json") 

    # MSP = MinkowskiSumProblem.from_json("instances/problems/prob-3-100|100-ll-2_1.json") 


    # Y = methods.MS_sum(MSP.Y_list)
    print(f"{MSP}")
    filter_time = time.time()
    Yn = methods.MS_sequential_filter(MSP.Y_list)
    print(f"{Yn.statistics=}")
    Yn.statistics['filter_time'] = time.time() - filter_time

    print(f"{Yn.statistics=}")
    print_timeit(0.1)

    # plt.show()
      
#     print(f"{MSP=}")
    # Yn = methods.MS_sequential_filter(MSP.Y_list, filter_alg = methods.KD_filter)
    # print_timeit()
    # print(f"{len(Yn)=}")
    # reset_timeit()
    # Yn = methods.MS_sequential_filter(MSP.Y_list, filter_alg = methods.naive_filter)
    # print_timeit()


def json_files_to_csv():

    with open('instances/results/algorithm1/results.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter = ",")
        
        # writer.writerow(";".join(("problem", "dim", "|Yn|")))
        statistics_keys = ["p", "supported", "extreme", "unsupported"]
        writer.writerow(("problem", "|Yn|", "|Y|", *statistics_keys ))

        for json_file in os.listdir("instances/results/algorithm1/"):
            if json_file.split(".")[-1] != "json": continue
            json_problem_file = json_file.split("alg1-")[1]
            print(f"{json_file=}")
            Yn = PointList.from_json( "instances/results/algorithm1/" + json_file)
            statistics = list(Yn.statistics.values())
            statistics_values =  [Yn.statistics[key][0] for key in statistics_keys]
            MSP = MinkowskiSumProblem.from_json("instances/problems/" + json_problem_file)
            row_data = [json_problem_file, str(len(Yn)), len(methods.MS_sum(MSP.Y_list).removed_duplicates()), *statistics_values]
            # row = ";".join(row_data)
            writer.writerow(row_data)



def remaining_instances():

    all_problems = os.listdir("instances/problems/")
    all_problems = sorted(all_problems, key = name_dict_keys )

    not_solved = []
    solved = []

    save_prefix = "alg1-"

    for p in all_problems:
        if save_prefix + p in os.listdir("instances/results/algorithm1/"):
            solved.append(p)
        else:
            not_solved.append(p)


    print(f"|solved| = {len(solved)}")
    print(f"|not solved| = {len(not_solved)}")

    # for p in not_solved:
        # print(f"{name_dict(p)}")

if __name__ == "__main__":
    remaining_instances()
    # test_times()
    # example_Yn()


    # main()
    # json_files_to_csv()

    # sorted_problems()
