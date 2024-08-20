import timing
import time
import logging

from timing import timeit, print_timeit, reset_timeit, terminate_after_x_minutes, log_every_x_minutes


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


def test_termination():

	# test correct returned value
    a_long_process_terminate = terminate_after_x_minutes(100 )(a_long_process)
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

def test_log_every_x():
    a_long_process_log = log_every_x_minutes(1*(1/60),logger)(a_long_process)

    res = a_long_process_log()
    assert res == True

	# make sure it termination does no have to wait for loginterval
    start_time = time.time()
    a_long_process_log = log_every_x_minutes(6*(1/60),logger)(a_long_process)
	a_long_process_log()
    total_time = time.time() - start_time
    assert total_time < 6 # Should take 5 + overhead
    
    

def test_terminate_and_log_combined():

    a_long_process_terminate = terminate_after_x_minutes(4*(1/60), logger)(a_long_process)
    a_long_process_log_and_terminate = log_every_x_minutes(1*(1/60), logger)(a_long_process_terminate)
    res = a_long_process_log_and_terminate()
    assert res is None

    a_long_process_terminate = terminate_after_x_minutes(100*(1/60), logger)(a_long_process)
    a_long_process_log_and_terminate = log_every_x_minutes(1*(1/60), logger)(a_long_process_terminate)
    res = a_long_process_log_and_terminate()
    assert res is True 

