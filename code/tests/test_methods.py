import pytest
from classes import Point, PointList, LinkedList, MinkowskiSumProblem, MSPInstances
import methods
from methods import U_dominates_L, induced_UB, lex_sort, N
from timing import timeit, print_timeit, set_defaults
from functools import reduce
import matplotlib.pyplot as plt
import copy

import collections

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
    assert Y == Y_copy
    Yn10 = methods.nondomDC_wrapper(Y)
    print(f"{Yn10=}")
    assert Yn1 == Yn2
    assert Yn2 == Yn3
    assert Yn3 == Yn4
    assert Yn6 == Yn7
    assert Yn7 == Yn8
    assert Yn8 == Yn9
    assert Yn9 == Yn10

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


def test_sequential():

    jsonfilename = "instances/problems/prob-2-100|100-mm-2_2.json"
    MSP = MinkowskiSumProblem.from_json(jsonfilename)

    Yn = methods.MS_sequential_filter(MSP.Y_list)
    Yn2 = methods.MS_sequential_filter(MSP.Y_list, N=methods.naive_filter)
    
    assert Yn == Yn2

def test_raw_write():
    MSP = MinkowskiSumProblem.from_json('./instances/problems/prob-3-100|100|100|100|100-mmmmm-5_3.json')
    Y = reduce(lambda x,y: x+y, MSP.Y_list[:2])
    # Y = Y + MSP.Y_list[-1]
    # print(f"{len(Y)=}")
    out_file = fr"/Users/au618299/Desktop/cythonTest/nondom/pointsCin" # c script directory
    Y.save_raw(out_file)
    # print(f"saved")
    # Yn = reduce(lambda x,y: methods.naive_filter(x+y), MSP.Y_list)
    # print(f"{len(Yn)=}")

def test_raw_read():
    in_file = filepath = fr"/Users/au618299/Desktop/cythonTest/nondom/pointsCin" # c script directory
    Y = PointList.from_raw(in_file)
    print(f"{Y=}")
    print(f"{Y.statistics=}")




def test_python_c_wrapper():
    MSP_list = [
                './instances/problems/prob-2-50|50-mm-2_3.json',
                './instances/problems/prob-3-50|50-mm-2_3.json',
                './instances/problems/prob-4-50|50-ll-2_3.json',
            ]
    for i, MSP_name in enumerate(MSP_list):
        print(f"{MSP_name=}")
        MSP = MinkowskiSumProblem.from_json(MSP_name)
        if i == 0:
            MSP.Y_list = [Y*(1**10) for Y in MSP.Y_list] # check large values
        if i == 1:
            MSP.Y_list = [Y*(1/2) for Y in MSP.Y_list] # check small values

        print(f"{i=}")
        Y = reduce(lambda x,y: x+y, MSP.Y_list)
        Y = PointList(Y.points)
        # YnDC = reduce(lambda x,y: methods.nondomDC_wrapper(x+y), MSP.Y_list)
        YnDC = methods.nondomDC_wrapper(Y)
        print(f"{len(YnDC)=}")
        Yn = methods.naive_filter(Y)
        # Yn = reduce(lambda x,y: methods.KD_filter(x+y), MSP.Y_list)
        print(f"{len(Yn)=}")
        print(f"{len(methods.N(Yn))=}")
        # print(f"{len(methods.nondomDC_wrapper(Yn))=}")

        # Y =reduce(lambda x,y: x+y, MSP.Y_list)
        

        for yn in YnDC:
            assert yn in Y
            assert yn in Yn
        for yn in Yn:
            res = Yn.dominates_point(yn)
            if res:
                print(f"{yn=}")
                print(f"{res=}")
                print(f"{res in Yn=}")
                print(f"{res in YnDC=}")
            assert not res
        for yn in Yn:
            assert yn in YnDC

        # Y.plot('Y')
#         plt.cla()
        # fig = plt.figure()
        # ax= plt.axes(projection = '3d')
        
        assert len(Yn) == len(YnDC)
        assert Yn == YnDC

        # YnDC.plot("DC", ax=ax)
        # Yn.plot("Yn",  ax = ax, marker='x')
        # # plt.show()
        # plt.cla()

def test_duplicates_filtered():

    # test if duplicates are removed
    # 2d
    Y = PointList([(1,1) for _ in range(10)] + [(1,2)])
    assert len(N(Y)) == 1
    # 3d
    Y = PointList([(1,1,1) for _ in range(10)] + [(2,1,0)])
    assert len(N(Y)) == 1

    Yn = methods.nondomDC_wrapper(Y)
    assert len(Yn) == 1
    assert Yn == PointList(((1,1),))

def test_c_wrapper_ND_pointsSum2():



    # test duplicates
    
    # jsonfilename = "instances/problems/prob-2-100|100-mm-2_2.json"
    # MSP = MinkowskiSumProblem.from_json(jsonfilename)

    MSP = list(MSPInstances(preset='algorithm2_test'))[0]
    A, B, C = MSP.Y_list
    
    ABn = methods.ND_pointsSum2_wrapper(A,B)
    ABn_check = methods.N(A+B)

    print(f"{[len(x) for x in (A,B,C)]=}")
    print(f"{len(ABn)=}")

    A.plot('A')
    B.plot('B', marker='x')
    

    ABn.plot('ABn')
    ABn_check.plot('ABn_check', marker = 'x')
    
    
    ABn_counts = collections.Counter(ABn)  
    ABn_check_counts = collections.Counter(ABn_check)  

    plottet = set()
    for y in ABn:
        if y in plottet:
            continue
        else:
            label = f"({ABn_counts[y]},{ABn_check_counts[y]})"
            y.plot(l=label, label_only=True)
            plottet.add(y)

    print(f"{ABn_counts.most_common(3)=}")
    print(f"{ABn_check_counts.most_common(3)=}")
    # ABn_counts.most_common(5)
    
    plt.show()
    
    assert ABn_check == ABn

    return

    jsonfilename = "instances/subproblems/sp-4-300-m_1.json"
    A = PointList.from_json(jsonfilename)
    
    jsonfilename = "instances/subproblems/sp-4-300-m_2.json"
    B = PointList.from_json(jsonfilename)

    save_path = "/Users/au618299/Desktop/cythonTest/nondom/temp"

    ABn = methods.ND_pointsSum2_wrapper(A,B)
    
    if False:
        A.plot('A')
        B.plot('B')
        ABn.plot('(A+B)n', SHOW=True)

    assert methods.N(A+B) == ABn

    # reverse
    ABn = methods.ND_pointsSum2_wrapper(B,A)

    assert methods.N(A+B) == ABn


    methods.call_c_ND_pointsSum2 = set_defaults(max_gb = 3)(methods.call_c_ND_pointsSum2)
    ABn = methods.ND_pointsSum2_wrapper(A,B)

    assert methods.N(A+B) == ABn

    
    # test forced termination time
    methods.call_c_ND_pointsSum2 = set_defaults(max_gb = 0.5, max_time = 0.1 * (1/60))(methods.call_c_ND_pointsSum2)
    
    ABn = methods.ND_pointsSum2_wrapper(A,B)
    assert ABn is None

    # test forced termination memory unsifficient
    methods.call_c_ND_pointsSum2 = set_defaults(max_gb = 0.0001, max_time = 200)(methods.call_c_ND_pointsSum2)
    
    ABn = methods.ND_pointsSum2_wrapper(A,B)
    assert ABn is None


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



def test_U_dominates_L():

    SHOW = False 
    def setup_instances():
        L_Y_val = list()

        # instances
        L = lex_sort(PointList.from_json('instances/subproblems/sp-2-10-l_1.json'))*4 + PointList(Point((10000,0)))
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')*2
        L_Y_val.append((L,Y,True))


        L = lex_sort(PointList.from_json('instances/subproblems/sp-2-10-l_1.json'))*3
        L = PointList([l for l in L] + [Point((5000,15000))])
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')*2
        L_Y_val.append((L,Y,False))

        L = lex_sort(PointList.from_json('instances/subproblems/sp-2-10-l_1.json'))*3
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')*2
        L_Y_val.append((L,Y,False))


        L = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')*1
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')*2
        L_Y_val.append((L,Y,False))



        # L_Y_val = list()
        # L = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')[::2] + PointList(Point((1000,0)))
        L = PointList(((5000,2000), (6000, 1000)))
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')
        L_Y_val.append((L,Y,False))

        L = PointList(((5000,2000), (6000, 1000))) + PointList(Point((0000,2000)))
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')
        L_Y_val.append((L,Y,True))

        L = PointList(((500,2000), (6000, 1000))) + PointList(Point((10000,-1500)))
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')
        L_Y_val.append((L,Y,False))

        L = PointList(((500,2000), (6000, 1000))) + PointList(Point((10000,1500)))
        Y = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')
        L_Y_val.append((L,Y,True))


        if False:
            L_Y_val = list()
            Y1 = PointList.from_json('instances/subproblems/sp-2-100-u_1.json')
            Y1se = PointList([l for l in Y1 if l.cls =='se'])
            Y2 = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')
            Y2se = PointList([l for l in Y2 if l.cls =='se'])

            U = N(Y1se + Y2se)
            
            Y1.plot(SHOW=False)
            Y2.plot(SHOW=False)
            Y1se.plot(SHOW=False, marker='x')
            Y2se.plot(SHOW=SHOW, marker='x')

            for y1 in lex_sort(Y1):
                print(f"{type(y1)=}")
                print(f"{type(Y2se)=}")
                L = Y2se + PointList(y1)

                L_Y_val.append((L,U,True))
                


        return L_Y_val

    for L, Y, supervised_answer in setup_instances():
        ######################## Figure 2d_lb_dominance START ########################
        fig_name = "2d_lb_dominance"
        print(f"")
        print(f"Plotting figure: {fig_name}")
        # define new figure
        fig, ax = plt.subplots(layout='constrained')
        Y = N(Y)
        L = N(L)
        L_is_dominated = U_dominates_L(Y,L)
        
        # sort for plot
        L = lex_sort(L)

        U = induced_UB(Y, assumption='consecutive', line=True)
        localNadir = induced_UB(Y, assumption='nonconsecutive', line=False)
        
        L.plot(f"$L$")
        L.plot(line=True, color = L.plot_color)
        Y.plot(f"$Y$", )
        
        for n in U:
            n.plot_cone(ax=ax, quadrant=1)

        print(f"{L_is_dominated=}")
        print(f"{supervised_answer=}")
        # assert L_is_dominated == supervised_answer

        U.plot(f"$U$", SHOW=False, line=True, color = 'red')
        localNadir.plot(f"$localNadir$",SHOW=SHOW,point_labels=True, color = 'red', marker = 'x')
        # Y.plot(f"$Y$",SHOW=SHOW, marker='x', color = 'red')
        plt.close()

        ######################### Figure 2d_lb_dominance END #########################


