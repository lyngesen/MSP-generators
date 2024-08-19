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
import threading 
import multiprocessing

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

def print_timeit(tolerance = 0):
    global TIME_dict
    # sort and show TIME_dict
    hline = 70
    print("_"*hline)
    TIME_dict = {k: v for k, v in sorted(TIME_dict.items(), key=lambda item: item[1])}
    for k,v in TIME_dict.items():
        calls = COUNT_dict[k]
        # if v > 0.01:
        if calls != 0 and v > tolerance:
            calls = f"{calls:13.2e}" if calls > 1_000_000 else f"{calls:13}" 
            print(f" {k:27} : {v:10.2f} seconds {calls} calls")
    print("_"*hline)
    print(f" {'Total time ':27} : {time.perf_counter() - START_TIME:10.2f} seconds")
    print("_"*hline)

    # TIME_dict_return = {k: v for k, v in TIME_dict.items() if COUNT_dict[k]>0}
    return TIME_dict

def reset_timeit():
    global TIME_dict
    global COUNT_dict
    global START_TIME

    TIME_dict = {k : 0 for k in TIME_dict.keys()}
    COUNT_dict = {k : 0 for k in COUNT_dict.keys()}
    START_TIME = time.perf_counter()


def time_object(object_name, prefix = None):
    """
    Modifies the incoming object_name (class or module) by adding the timeit_wrapper to each callable in the object.

    exceptions include '__class__', '__new__', '__getattribute__' and names which include 'recursion'
    """
    if prefix == None:
        prefix = object_name.__name__

    for fct_str in dir(object_name):
        fct = getattr(object_name, fct_str)
        if callable(fct) and fct_str not in ['__class__', '__new__', '__getattribute__'] and 'recursion' not in fct_str:
            fct = timeit(fct, f'{prefix}.{fct_str}')
            setattr(object_name, fct_str, fct)
    return object_name






def log_every_x_minutes(x, logger):
    """
    A decorator for creating a log every x minutes. Threading is required as the process has to run paralell to calling the function func
    

    Usage:
        import logging
        logging.basicConfig(level=logging.INFO, filename=logname)
        logger = logging.getLogger(logname)

        @log_every_x_minutes(30, logger)
        def example_function_which_takes_a_long_time_to_run():
            time.sleep(a long time)
    (or)
        example_fct = log_every_x_minutes(30, logger)(example_fct)


    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            def log_time():
                while True:
                    elapsed_time = time.time() - start_time
                    logger.info(f"The function {func.__name__} has been running for {elapsed_time / 60:.2f} minutes")
                    time.sleep(x * 60)
            
            # Start the logging thread
            log_thread = threading.Thread(target=log_time)
            log_thread.daemon = True
            log_thread.start()
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def terminate_after_x_minutes(x, logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start the function as a process
            p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
            p.start()

            # Wait for the specified time or until the process finishes
            p.join(timeout=x*60)

            # If the process is still active after the wait time
            if p.is_alive():
                print(f"{func.__name__} is running... killing process after {x:.2f} minutes {x/60:.2f} hours")
                if logger:
                    logger.info(f"{func.__name__} is running... killing process after {x/60:.2f} seconds {x:.2f} minutes {x*60:.2f} hours")
                    logger.warning(f"{func.__name__} is running... killing process after {x*60:.2f} seconds {x:.2f} minutes {x/60:.2f} hours")

                # Terminate the process
                p.terminate()

                # Ensure the process has terminated
                p.join()
        return wrapper
    return decorator
