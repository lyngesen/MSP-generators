import timing



from timing import timeit, print_timeit, reset_timeit

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
       
     
