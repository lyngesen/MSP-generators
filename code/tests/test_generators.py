import random
from generator import generate_PointList

def test_generators():
    random.seed(1)
    S = 2
    P = [1,2,3]
    P = [2,3]
    N = [10,100,1_000]
    methods = ["DISK","BINOM","CONCAVE"]
    for p in P:
        for n in N:
            for s in range(1,S+1): 
                for method in methods:
                    Y = generate_PointList(n,p, method = method)
                    fname = f"instances/testsets/{method}-p{p}-n{n}-s{s}"


