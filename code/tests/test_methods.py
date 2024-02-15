import pytest
from classes import Point, PointList, LinkedList
import methods
from timing import timeit, print_timeit
from functools import reduce
import matplotlib.pyplot as plt
import copy

def test_sorting():
    testset = "instances/testsets/BINOM-p2-n100-s1"
    Y = PointList.from_csv(testset)
    YS = methods.lex_sort(Y)
    Y = PointList.from_csv(testset)
    YS2 = methods.lex_sort_linked(Y)

    assert YS == YS2


    Y = PointList(((1,2,3),(2,2,2),(3,2,1), (1,1,2)))
    Y_lex = PointList(((1,1,2), (1,2,3),(2,2,2),(3,2,1)))
    # cast to list and check sorting sequence returned by lex_sort is correct
    assert list(methods.lex_sort(Y).points) == list(Y_lex.points)




    
def test_single():

    Y = PointList(((2,7), (3,9), (4,3), (5,8), (7,5), (6,4), (8,6), (9,2)))
    methods.two_phase_filter(Y)

    # single set test
    testset = "instances/testsets/DISK-p2-n10-s1"
    Y = PointList.from_csv(testset)
    Y_copy = copy.deepcopy(Y)
    assert Y == Y_copy
    Yn1 = methods.naive_filter(Y, MCtF = False)
    assert Y == Y_copy
    Yn2 = methods.naive_filter(Y, MCtF = True)
    assert Y == Y_copy
    Yn3 = methods.unidirectional_filter(Y)
    assert Y == Y_copy
    Yn4 = methods.lex_filter(Y)
    assert Y == Y_copy
    Yn6 = methods.basic_filter(Y)
    assert Y == Y_copy
    Yn7 = methods.KD_filter(Y)
    assert Y == Y_copy
    Yn8 = methods.KD_filter(Y)
    assert Y == Y_copy
    Yn9 = methods.two_phase_filter(Y)
    assert Yn1 == Yn2
    assert Yn2 == Yn3
    assert Yn3 == Yn4
    assert Yn6 == Yn7
    assert Yn7 == Yn8
    assert Yn8 == Yn9

    # Check input PointList unchanged by filtering algorithms
    assert Y == Y_copy
    # test sort 
    Y_lex = methods.lex_sort(Y)
    for i in range(len(Y_lex)-1):
        assert Y_lex[i].lex_le(Y_lex[i+1])
        assert not Y_lex[i] > Y_lex[i+1]
    Y.print_data()
    Y_lex.print_data()
    Yn1.print_data()
    assert Y_lex == Y

    assert Y_lex == methods.lex_sort_linked(Y)

    # test plot
    Y_lex.plot(l='lex_sorted')



    # test no Yn duplicates
    Y = PointList(((2,7), (3,9), (4,3), (5,8), (7,5), (6,4), (8,6), (9,2),
                   (2,7), (3,9), (4,3), (5,8), (7,5), (6,4), (8,6), (9,2)))
    Yn = PointList(((2,7),(4,3), (9,2)))
    assert methods.N(Y) == Yn
    assert methods.naive_filter(Y) == Yn
    assert methods.two_phase_filter(Y) == Yn



def test_MS(): 
    # MS test
    testset1 = "instances/testsets/BINOM-p2-n100-s1"
    # Y1 = PointList.from_csv(testset)
    testset2 = "instances/testsets/CONCAVE-p2-n100-s1"
    # Y2 = PointList.from_csv(testset)
    testset3 = "instances/testsets/BINOM-p2-n20-s2"
    # Y3 = PointList.from_csv(testset)
    # Y4 = PointList.from_csv(testset)
    Y_list = tuple([PointList.from_csv(Y) for Y in (testset1, testset2, testset3) ])

    Y_ms = methods.MS_sum(Y_list)

    # Y_ms.plot(l = 'Y_MS')

    Y_list = tuple([PointList.from_csv(Y) for Y in (testset1, testset2, testset3) ])
    Yn = methods.naive_filter(Y_ms, True)
    Y_list = tuple([PointList.from_csv(Y) for Y in (testset1, testset2, testset3) ])
    Yn1 = methods.MS_naive_filter(Y_list)
    Y_list = tuple([PointList.from_csv(Y) for Y in (testset1, testset2, testset3) ])
    Yn2 = methods.MS_doubling_filter(Y_list)
    Y_list = tuple([PointList.from_csv(Y) for Y in (testset1, testset2, testset3) ])
    Yn3 = methods.MS_sequential_filter(Y_list)

    assert Yn == Yn1
    assert Yn == Yn2
    assert Yn == Yn3


def test_find_generator():

    Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s1")
    Y2 = PointList.from_csv("instances/testsets/CONCAVE-p2-n10-s2")
    Uc = methods.find_generator_U(Y1,Y2)

def test_induced_UB():

    Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n10-s2")
    Y2 = PointList.from_csv("instances/testsets/DISK-p2-n10-s1")
    U = methods.induced_UB(Y1)
    UL = methods.induced_UB(Y1, line=True)

