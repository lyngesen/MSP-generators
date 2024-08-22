import timing
import time
import logging

from classes import MinkowskiSumProblem, MSPInstances

import methods 

from timing import timeit, print_timeit, reset_timeit, terminate_after_x_minutes, log_every_x_minutes, terminate_and_log, set_defaults


logger = logging.getLogger('pytest_log')

def test_timing():
    @timeit
    def blabla():
        s = 0
        for _ in range(10000000):
            s += _**2

    def blibli():
        s = 0
        for _ in range(1000000):
            s += _**2

    blibli = timeit(blibli)


    blabla()
    blibli()
    blabla()
    blibli()
    blibli()
    
    print_timeit()
    reset_timeit()
    print_timeit()
       
 
A_LONG_TIME = 5
def a_long_process():
    for _ in range(A_LONG_TIME):
        time.sleep(1)
    
    return True
A_SHORT_TIME = 1
def a_short_process():
    for _ in range(A_SHORT_TIME):
        time.sleep(1)
    
    return True


def test_termination():

	# test correct returned value
    a_long_process_terminate = terminate_after_x_minutes(10 * (1/60) )(a_long_process)
    res = a_long_process_terminate()
    assert res == True

	# test termination works, and None is returned
    a_long_process_terminate = terminate_after_x_minutes(2 * (1/60))(a_long_process)
    res = a_long_process_terminate()
    print(f"{res=}")

    assert res is None

	# test logger argument
    a_long_process_terminate_and_log = terminate_after_x_minutes(1 * (1/60), logger)(a_long_process)
    res = a_long_process_terminate_and_log()

    print(f"{res=}")

	# make sure it termination does no have to wait for loginterval
    start_time = time.time()
    a_long_process_term = terminate_after_x_minutes(6*(1/60),logger)(a_long_process)
    a_long_process_term()
    total_time = time.time() - start_time
    assert total_time < 6+1 # Should take 5 + overhead



def old_test_log_every_x():
    a_long_process_log = log_every_x_minutes(1*(1/60),logger)(a_long_process)

    res = a_long_process_log()
    assert res == True

	# make sure it termination does no have to wait for loginterval
    start_time = time.time()
    a_long_process_log = log_every_x_minutes(6*(1/60),logger)(a_long_process)
    a_long_process_log()
    total_time = time.time() - start_time
    assert total_time < 6 # Should take 5 + overhead
    
    

def old_test_terminate_and_log_combined():

    a_long_process_terminate = terminate_after_x_minutes(4*(1/60), logger)(a_long_process)
    a_long_process_log_and_terminate = log_every_x_minutes(1*(1/60), logger)(a_long_process_terminate)
    res = a_long_process_log_and_terminate()
    assert res is None

    a_long_process_terminate = terminate_after_x_minutes(10*(1/60), logger)(a_long_process)
    a_long_process_log_and_terminate = log_every_x_minutes(1*(1/60), logger)(a_long_process_terminate)
    res = a_long_process_log_and_terminate()
    assert res is True 

def old_test_terminate_and_log():
    
    # no log
    a_long_process_decorated = terminate_and_log(max_time = 4*(1/60),  logger = logger)(a_long_process)
    res = a_long_process_decorated()
    assert res is None

    # both - natural termination
    a_long_process_decorated = terminate_and_log(max_time = 6*(1/60), log_interval = 1*(1/60), logger = logger)(a_long_process)
    res = a_long_process_decorated()
    assert res is True 
    
    # both - natural termination - large max_time
    a_long_process_decorated = terminate_and_log(max_time = 100*(1/60), log_interval = 1*(1/60), logger = logger)(a_long_process)
    res = a_long_process_decorated()
    assert res is True 
    
    # both - natural termination - log_interval max_time
    a_long_process_decorated = terminate_and_log(max_time = 100*(1/60), log_interval = 100*(1/60), logger = logger)(a_long_process)
    res = a_long_process_decorated()
    assert res is True 

    # both - forced termination
    a_long_process_decorated = terminate_and_log(3*(1/60), 1*(1/60), logger)(a_long_process)
    res = a_long_process_decorated()
    assert res is None 


def old_test_terminate_and_log_timing():

	# make sure it termination does no have to wait for loginterval
    start_time = time.time()
    a_long_process_decorated = terminate_and_log(5*(1/60),False, logger)(a_long_process)
    a_long_process_decorated()
    total_time = time.time() - start_time
    assert total_time < 6 # Should take 5 + overhead
    
    
	# make sure it termination does no have to wait for loginterval
    start_time = time.time()
    a_long_process_decorated = terminate_and_log(5*(1/60),1*(1/60), logger)(a_long_process)
    a_long_process_decorated()
    total_time = time.time() - start_time
    assert total_time < 6


	# natural termination not affected
    start_time = time.time()
    a_long_process_decorated = terminate_and_log(30*(1/60),10*(1/60), logger)(a_long_process)
    a_long_process_decorated()
    a_long_process_decorated()
    total_time = time.time() - start_time
    assert total_time < (A_LONG_TIME+1)*2 + 1


def old_test_terminate_and_log_timing_short():
    logger.info(f"natural termination not affected 1")
	# natural termination not affected
    start_time = time.time()
    a_short_process_decorated = terminate_and_log(30*(1/60),10*(1/60), logger)(a_short_process)
    N = 2
    for n in range(N):
        print(f"{n=}")
        a_short_process_decorated()
    total_time = time.time() - start_time
    assert total_time < (A_SHORT_TIME+1)*N + 1


    # calculate nessesary time
    test_problem = './instances/problems/prob-3-100|100|100-mmm-3_1.json'
    required_time = time.time()
    for n in range(N):
        MSP = MinkowskiSumProblem.from_json(test_problem)
        methods.MS_sequential_filter(MSP.Y_list)
    required_time = time.time() - required_time
    logger.info(required_time)


    # natural termination not affected
    logger.info(f"natural termination not affected 2")
    start_time = time.time()
    # decorated_filter = terminate_and_log(10*(1/60),5*(1/60), logger)(methods.MS_sequential_filter)
    decorated_filter = terminate_and_log(10*(1/60),5*(1/60), logger)(methods.MS_sequential_filter)
    for n in range(N):   
        MSP = MinkowskiSumProblem.from_json(test_problem)
        decorated_filter(MSP.Y_list)

    total_time = time.time() - start_time
    assert total_time < required_time + 1 # + overhead



    # calculate nessesary time
    required_time = time.time()
    for MSP in MSPInstances(preset = 'grendel_test'): 
        methods.MS_sequential_filter(MSP.Y_list)
    required_time = time.time() - required_time
    logger.info(required_time)

    # natural termination not affected
    logger.info(f"natural termination not affected 3")
    start_time = time.time()
    decorated_filter = terminate_and_log(30*(1/60),10*(1/60), logger)(methods.MS_sequential_filter)
    for MSP in MSPInstances(preset = 'grendel_test'):
        decorated_filter(MSP.Y_list)
    total_time = time.time() - start_time
    assert total_time < required_time + 1 # + overhead

 
def test_set_defaults():
    N = 2

    # calculate nessesary time
    test_problem = './instances/problems/prob-3-100|100|100-mmm-3_1.json'
    required_time = time.time()
    for n in range(N):
        MSP = MinkowskiSumProblem.from_json(test_problem)
        methods.MS_sequential_filter(MSP.Y_list)
    required_time = time.time() - required_time
    logger.info(required_time)



    methods.call_c_nondomDC = set_defaults(max_time = 6 * (1/60))(methods.call_c_nondomDC)
    
    # calculate nessesary time
    decorator_time = time.time()
    for n in range(N):
        MSP = MinkowskiSumProblem.from_json(test_problem)
        methods.MS_sequential_filter(MSP.Y_list)
    decorator_time = time.time() - decorator_time
    logger.info(decorator_time)

    assert decorator_time < required_time + 1

