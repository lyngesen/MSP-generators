''' script for answering research question 1:
RQ1: How large is |Yn| given |Yn^s| for s in S

For each problem MSP (from instances/problem)
    Find Yn and save solution (PointList) to json file and add statistics with |Yn|
'''

from classes import Point, PointList, MinkowskiSumProblem, KD_Node, KD_tree, MSPInstances
import methods
from algorithm2 import algorithm2
import minimum_generator
import timing
from timing import timeit, print_timeit, reset_timeit, time_object, log_every_x_minutes, terminate_and_log, set_defaults
from methods import N
# public library imports
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import os
import csv
import time
import itertools
import argparse

# for logging
import logging

algorithm2 = timeit(algorithm2)

time_object(KD_tree)
time_object(KD_Node)
time_object(PointList)
time_object(Point)
time_object(timing)
time_object(minimum_generator, 'PYOMO')
time_object(MinkowskiSumProblem,'MSP')
time_object(methods, prefix ="ALG")


def name_dict(problem_file):
    filename = problem_file
    problem_file = problem_file.split(".json")[0]
    problem_file, seed = problem_file.split("_")
    _, p, size, method, M = problem_file.split("-")
    size = size.split("|")[0]
    p, M, size, seed = int(p), int(M), int(size), int(seed)
    D = {'filename': filename, 'p': p, 'method':method, 'M': M, 'size': size, 'seed':seed}
    return D
    
def name_dict_keys(problem_file):
    D = name_dict(problem_file)
    return ( D['M'],D['size'], D['p'], D['seed'])

# def sorted_problems():
    # all_problems = os.listdir("instances/problems/")

    # print(f"{name_dict(all_problems[0])=}")

    # all_problems = sorted(all_problems, key = name_dict_keys )

    # for p in all_problems[:100]:
        # # print(f"{name_dict(p)}")
        # print(" ".join(f"{k} = {v},\t" for k, v in name_dict(p).items()))
    # # print(f"{all_problems[:10]=}")

def main():
    
    # run algorithm 1 on specified test instances
    m_options = (2,3,4,5) # subproblems
    p_options = (2,3,4,5) # dimension
    generation_options = ['m','u'] # generation method
    size_options = (50, 100, 200, 300) # subproblems size
    seed_options = [1,2,3,4,5]
    

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
    tosolve = 0
    for problem_name in all_problems:
        # print(" ".join(f"{k} = {v},\t" for k, v in name_dict(problem_name).items()))

        if problem_name.split(".")[-1] != "json": continue
        if save_prefix + problem_name in os.listdir("instances/results/algorithm1/"):
            # print(f"  problem solved - skipping {problem_name}")
            continue
        MSP = MinkowskiSumProblem.from_json("instances/problems/"+ problem_name) 

        # if MSP.dim == 2:
            # continue

        # if len(MSP.Y_list[0]) == 50:
            # continue

        instance_dict = name_dict(problem_name)
        if not all((instance_dict['p'] in p_options,
               instance_dict['M'] in m_options,
               set(instance_dict['method']).issubset(set(generation_options)),
               instance_dict['size'] in size_options,
               instance_dict['seed'] in seed_options
               )):
            continue

        else:
            print(f"solvinf instance {problem_name=}")

        if get_Y_max_size(MSP) > 100_000_000:
            # print(f"  problem too big (skipping): {get_Y_max_size(MSP)=}")
            print(f"  problem very large (maybe skip): {get_Y_max_size(MSP)=}")
            # continue


        tosolve +=1
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


    print(f"{tosolve=}")
def test_times():


    MSP_list= [
               # "./instances/problems/prob-5-100|100-ll-2_1.json",
               # "./instances/problems/prob-4-100|100-ll-2_2.json",
               #"./instances/problems/prob-5-50|50|50-lll-3_1.json",
               # "./instances/problems/prob-3-50|50|50|50-uuuu-4_1.json",
               # "./instances/problems/prob-5-600|600|600-uuu-3_1.json",
               # "./instances/problems/prob-4-100|100|100-mmm-3_1.json",
               "./instances/problems/prob-4-100|100|100|100-mmmm-4_1.json",
               # "./instances/problems/prob-5-50|50|50|50-mmmm-4_1.json",
               ]

    for MSP_name in MSP_list:
        MSP = MinkowskiSumProblem.from_json(MSP_name)
        csv_file_path = 'alg1_grendel.csv'
        print(f"{MSP}")
        filter_time = time.time()
        Yn = methods.MS_sequential_filter(MSP.Y_list)
        Yn.statistics['filter_time'] = time.time() - filter_time
        print(f"{len(Yn)=}")
        time_dict = print_timeit(0.1)
        data = {'problem': MSP.filename}
        data.update(Yn.statistics)
        # data.update(time_dict)
#         with open(csv_file_path, 'a') as csv_file:
            # # add header if file empty
            # writer = csv.writer(csv_file)
            # if os.path.getsize(csv_file_path) == 0:
                # writer.writerow(data.keys())
            # writer.writerow(data.values())

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

    return not_solved





def algorithm1():
    # run algorithm 1 on specified test instances
    
    m_options = (2,3,4,5) # subproblems
    p_options = (2,3,4,5) # dimension
    generation_options = ['m','u'] # generation method
    # size_options = (50, 100, 150, 200,300, 600) # subproblems size
    size_options = (50, 100, 150, 200, 300) # subproblems size
    seed_options = [1,2,3,4,5]
    

    if False: # testing
        m_options = (3,) # subproblems
        p_options = (4,) # dimension
        size_options = (200,) # subproblems size
        generation_options = ['l'] # generation method


    not_solved = remaining_instances()
    not_solved_subset = list()
    to_solve = 0
    for filename in not_solved:
        instance_dict = name_dict(filename)
        if all((instance_dict['p'] in p_options,
               instance_dict['M'] in m_options,
               set(instance_dict['method']).issubset(set(generation_options)),
               instance_dict['size'] in size_options,
               instance_dict['seed'] in seed_options
               )):
            # print(f"{filename=}")
            not_solved_subset.append(filename)
            to_solve +=1
            
    print(f"{len(not_solved)=}")
    print(f"{to_solve=}")

    # solve each selected instance and save PointList as file
    print(f"{len(not_solved_subset)=}")

    not_solved_subset = sorted(not_solved_subset, key = name_dict_keys )
 

    print(f"{not_solved_subset=}")

    save_prefix = "alg1-"
    for filename in not_solved_subset:
        if True: return
        MSP = MinkowskiSumProblem.from_json("instances/problems/"+ filename) 

        print(f"{MSP=}")
        filter_time = time.time()
        Yn = methods.MS_sequential_filter(MSP.Y_list)
        Yn.statistics['filter_time'] = time.time() - filter_time

        Yn.save_json("instances/results/algorithm1/"  +  save_prefix + filename)
        print(f"{len(Yn)=}")

        print_timeit()
        reset_timeit()
        print(" ")
        # return 




def convert_all_raw_files():
    ''' ad hoc script for converting all pointLists saved as .raw files into .json files'''


    solution_dir = './instances/results/algorithm1/'
    out_dir = '/Users/au618299/Desktop/large-result-files-alg1/'
    raw_files = [file for file in os.listdir(solution_dir) if file.split(".")[-1]=='raw']

    with alive_bar(len(raw_files), enrich_print=True) as bar:
        for raw_file in raw_files:

            if raw_file.replace('.raw','.json') in os.listdir(out_dir):
                # print(f"{raw_file=} already in {out_dir=}")
                bar()
                continue

            print(f"{raw_file=}")
            dir_size_gb = sum(os.path.getsize(out_dir + f) for f in os.listdir(out_dir) if os.path.isfile(out_dir + f))/(1024**3)
            print(f"Current size of dir is {dir_size_gb:.2f} GB")
            Y = PointList.from_raw(solution_dir + raw_file)
            # print(f"{Y.statistics=}")
            Y.statistics = PointList.from_json(solution_dir + raw_file.replace('.raw','.json')).statistics
            # print(f"{Y.statistics=}")
            # Y = PointList.from_json('./instances/results/algorithm1/alg1-prob-2-100|100-ll-2_1.json') # for testing
            Y.save_json(out_dir + raw_file.replace('.raw','.json'), max_file_size = 1000)

            bar()


            




def main():

    TERMINATE_AFTER_X_MINUTES = 60
    MEMORY_LIMIT = 1 # GB
    save_prefix = 'alg1-'
    MSP_preset = 'algorithm1'

    # save_solution_dir = './instances/results/algorithm1/'
    save_solution_dir = './instances/results/testdir/'


    # parse arguments
    parser = argparse.ArgumentParser(description="Save instance results PointList in dir.")
    parser.add_argument('-timelimit', type=float, required=False, help='Time limit for each instance')
    parser.add_argument('-npartition', type=int, required=False, help='Total partitions (n) of test instances')
    parser.add_argument('-kpartition', type=int, required=False, help='Number of specific test intsance partition')
    parser.add_argument('-memorylimit', type=float, required=False, help='Memory limit for each instance')
    parser.add_argument('-outdir', type=str, required=False, help='Result dir, where instances are saved')
    parser.add_argument('-logpath', type=str, required=False, help='path where log (algorithm1.log) files are to be saved')
    parser.add_argument('-msppreset', type=str, required=False, help='Choice of preset instances to solve default: algorithm1. other choices grendel_test, algorithm2')
    parser.add_argument('-solveall', action='store_true', help='if flag added, all instances are solved (already solved instances will not be filtered out)')
    parser.add_argument('-alg2', action='store_true', help='if flag added, MGS will be solved using algorithm2)')


    args = parser.parse_args()
    outdir = args.outdir
    logpath = args.logpath
    if args.timelimit:
        TERMINATE_AFTER_X_MINUTES = args.timelimit
    if args.msppreset:
        MSP_preset = args.msppreset
    if args.memorylimit:
        MEMORY_LIMIT = args.memorylimit


    TI = MSPInstances(MSP_preset, ignore_ifonly_l=args.alg2) # if -alg2 then 'l' instances are ignored 

    if not args.solveall:
        TI.filter_out_solved(save_prefix, save_solution_dir)

    if args.npartition:
        assert args.kpartition is not None ,f'if partition flag is set, the specific partition must be specified -npartition => -kpartition{args.kpartition,args.npartition=}'
        assert args.kpartition < args.npartition, f"{args.kpartition,args.npartition=}"
        TI.partition(args.npartition, args.kpartition)
    


    if outdir:
        assert outdir[-1] =='/'
        save_solution_dir = outdir
        # print(f"\tDirectory path provided: {outdir}")
        # print(f"\t{os.path.exists(outdir)=}")

    # add logger
    logname = 'algorithm1.log'
    if logpath:
        logpath = logpath
    else: 
        logpath = logname
    logging.basicConfig(level=logging.INFO, 
                        filename=logpath,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        )
    logger = logging.getLogger(logname)


    options = {
        "args.solveall": args.solveall,
        "TERMINATE_AFTER_X_MINUTES": TERMINATE_AFTER_X_MINUTES,
        "MEMORY_LIMIT": MEMORY_LIMIT,
        "MSP_preset": MSP_preset,
        "outdir":outdir,
        "logpath":logpath,
        "running alg2":args.alg2,
        "n partitions": args.npartition,
        "k partition": args.kpartition,
    }

    options_str = "Options:\n" + "\n".join([f"\t{key}={value}" for key, value in options.items()])

    print(options_str)
    logger.info(options_str)


    # add decorators
    #   terminate after specified number of minutes
    #   limit memory used by c script
 
    
    methods.call_c_nondomDC = set_defaults(max_time = TERMINATE_AFTER_X_MINUTES, logger = logger)(methods.call_c_nondomDC)
    methods.call_c_ND_pointsSum2 = set_defaults(max_gb = MEMORY_LIMIT, logger=logger)(methods.call_c_ND_pointsSum2)

    print(f"{TI=}")
    logger.info(f"{TI=}")
    if len(TI.filename_list) < 10:
        print(f"{TI.filename_list=}")
    logger.info(f'Running algorithm1 on test instance set {TI}')

    # with alive_bar(len(TI.filename_list), enrich_print=True) as bar:
    if True:
        for i, MSP in enumerate(TI):
            time_start = time.time()

            status_msg = f"Solve status {i}/{len(TI.filename_list)} ({(i / len(TI.filename_list) * 100):.2f}%)"
            logger.info(f"{MSP}")
            logger.info(status_msg)
            print(f"{MSP}")
            print(status_msg)

            filter_time = time.time()


            Yn = methods.MS_sequential_filter(MSP.Y_list)
            if Yn is None: # if process was stopped
                logger.warning(f"instance {MSP=} terminated after {TERMINATE_AFTER_X_MINUTES} minutes")
                Yn = PointList()
                Yn.statistics['filter_time'] = time.time() - filter_time
                Yn.statistics['card'] = None

            else: 
                Yn.statistics['filter_time'] = time.time() - filter_time


            Yn.save_json(save_solution_dir + 'algorithm1/alg1-' + MSP.filename.split('/')[-1])
            logger.info(f"{MSP.filename=}, {len(Yn)=}, filter_time = {Yn.statistics['filter_time']}")
            
            if args.alg2: # also run alg2
                logger.info(f'Running alg2 for {MSP=}')
                MGS, Yn_2 = algorithm2(MSP, logger = logger)

                statistics_str = "stat2:\n" + "\n".join([f"\t{key}={value}" for key, value in MGS.statistics.items()])
                logger.info(statistics_str)

                Yn_2.statistics['filter_time'] = MGS.statistics['time_simplefilter']
                assert Yn_2 == Yn, f"{len(Yn),len(Yn_2)=}"
                Yn_2.save_json(save_solution_dir + 'algorithm1/alg2-' + MSP.filename.split('/')[-1])
                MGS.filename = save_solution_dir + 'algorithm2/MGS-' + MSP.filename.split('/')[-1]
                MGS.save_json(MGS.filename)
            

            if True:
                print_timeit(tolerance=5, logger=logger)
                reset_timeit()

            # print(" ")
            # bar()



if __name__ == "__main__":
    main()

    # convert_all_raw_files()

    # remaining_instances()
    # algorithm1()
    # test_times()
    # example_Yn()
    # main()
    # json_files_to_csv()
    # sorted_problems()
