from classes import Point, PointList, MinkowskiSumProblem, MSPInstances
import methods

from algorithm2 import algorithm2, alg2, algorithm2_args



def test_algorithm2_return():

    # TI = MSPInstances(preset='algorithm2_test', max_instances = 2)
    # MSP = prob-2-100|100-ll-2_2.json
    TI = MSPInstances(preset='algorithm2_test', max_instances = 2)
    for MSP in TI:
        print(f"{MSP=}")
        MGS, Yn = algorithm2(MSP)        

        print(f"{MGS.statistics=}")
        print(f"{len(Yn)=}")
        assert len(methods.N(methods.MS_sequential_filter(MGS.Y_list))) == len(Yn)
        assert methods.N(methods.MS_sequential_filter(MGS.Y_list)) == Yn

