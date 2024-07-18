from classes import Point, PointList, MinkowskiSumProblem, MSPInstances
import methods

from algorithm2 import algorithm2, alg2

# import pytest
# @pytest.mark.filterwarnings("ignore:invalid scape sequence*DepreciationWarning")
def test_algorithm2():

    for MSP in MSPInstances(preset='algorithm2_test'):

        print("running new alg2")
        Y_MGS = algorithm2(MSP)
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


