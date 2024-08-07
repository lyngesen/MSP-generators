from classes import Point, PointList, MinkowskiSumProblem, MSPInstances
import methods

from algorithm2 import algorithm2, alg2

# import pytest
# @pytest.mark.filterwarnings("ignore:invalid scape sequence*DepreciationWarning")
def test_algorithm2():

    TI = MSPInstances(preset='algorithm2_test', max_instances = 1)
    print(f"{TI=}")
    print(f"{TI.filename_list=}")
    for MSP in TI:

        print("running new alg2")
        Y_MGS, Yn_size = algorithm2(MSP)
        MGS_size = sum([len(Y) for Y in Y_MGS])
        print(f"{MGS_size=}")
        for i, g in enumerate(Y_MGS):
            print(f"|G{i+1}| = {len(g)}")
        
        print("running old alg2")

        Y_MGS_old = alg2(MSP)
        MGS_size_old = sum([len(Y) for Y in Y_MGS])
        print(f"{MGS_size=}")
        for i, g in enumerate(Y_MGS):
            print(f"|G{i+1}| = {len(g)}")

        assert MGS_size == MGS_size_old


