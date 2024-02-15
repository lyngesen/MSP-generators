"""
Decorator @timeit for timing running times and counting function calls

function print_timeit() for printing result

example use:

    from timing import timeit, print_timeit

    # Define function with decorator
    @timeit
    def blibli():
         s = 0
         for _ in range(1000000):
             s += _**2

    # Add decorator to already defined function 
    blibli = timeit(blibli)

    ...
    blibli()
    blibli()
    blabla()

    print_timeit()
    
    >> RESULT
    _______________________________________________________
     blibli               :  100.4 seconds       2 calls
     blabla               :  130.0 seconds       1 calls
    _______________________________________________________
     Total time           :  150.0 seconds
    _______________________________________________________


"""
from functools import wraps
import time


# Define global dictionaries
TIME_dict = {}
COUNT_dict = {}
START_TIME = time.perf_counter()

def timeit(func, keyname = None):
    if keyname == None:
        keyname = func.__name__
    TIME_dict[keyname] = 0
    COUNT_dict[keyname] = 0
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        TIME_dict[keyname] += total_time
        COUNT_dict[keyname] += 1
        #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def print_timeit():
    global TIME_dict
    # sort and show TIME_dict
    print("_"*55)
    TIME_dict = {k: v for k, v in sorted(TIME_dict.items(), key=lambda item: item[1])}
    for k,v in TIME_dict.items():
        if COUNT_dict[k]!= 0:
            print(f" {k:20} : {v:5.2f} seconds {COUNT_dict[k]:7} calls")
    print("_"*55)
    print(f" {'Total time ':20} : {time.perf_counter() - START_TIME:5.2f} seconds")
    print("_"*55)

def reset_timeit():
    global TIME_dict
    global COUNT_dict
    global START_TIME

    TIME_dict = {k : 0 for k in TIME_dict.keys()}
    COUNT_dict = {k : 0 for k in COUNT_dict.keys()}
    START_TIME = time.perf_counter()
