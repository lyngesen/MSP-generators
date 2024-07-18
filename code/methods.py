
"""
methods/algorithms for filtering out nondominated points
specifically for a single set (Non minkowski sum problem)

IMPLEMENTED methods:
    basic_filter(Y) -> Yn
    naive_filter(Y) -> Yn
    lex_sort(Y) -> sorted(Y)
    unidirectional_filter(Y) -> Yn

    MS_sum(Y_list) -> Y_ms, Minkowski sum of sets in Y_list
    MS_naive_filter(Y_list) -> N(Y_ms)
    MS_sequential_filter(Y_list) -> N(Y_ms)
    MS_doubling_filter(Y_list) -> N(Y_ms)
"""
from classes import Point, PointList, LinkedList, Node, KD_Node, KD_tree
import numpy as np
import os
import subprocess # for running c execute
from collections import deque # for fast leftappend
from operator import itemgetter # for lexsort function to define keys
import math


def basic_filter(Y:PointList):
    """
    input: PointList
    output: PointList with all nondominated points removed

    ALG:
        For each point y, check if any other point y2 dominates y: if no, add y to Yn
    """
    Yn = []

    for y in Y:
        for y2 in Y:
            if y2 < y:
               break 
        else:
            Yn.append(y)

    return(PointList(Yn))


def naive_filter(Y:PointList, MCtF = False) -> PointList:
    """
    input: PointList
    output: PointList with all nondominated points removed

    ALG:
        For each point y, check if any other point y2 dominates y: if no, add y to Yn
    """
    Yn = deque()

    for i, y in enumerate(Y):
        dominated_indices_y = set()
        for j, y2 in enumerate(Yn):
            if y2 <= y:
                break # discard y
            if y2 > y:
                dominated_indices_y.add(j) # record dominance
        else:
            # remove dominated points
            Yn = deque((y_ for j_, y_ in enumerate(Yn) if j_ not in dominated_indices_y))
            # add nondominated candidate
            if MCtF and dominated_indices_y:
                Yn.appendleft(y)
            else:
                Yn.append(y)

    # Y.points = Yn
    return(PointList(Yn))


def two_phase_filter(Y: PointList) -> PointList:
    """ Two phase filter from Chen2012 for filtering a list of points

    Args:
        Y (PointList): Yn (PointList) set of nondominated points 

    Returns: PointList with all nondominated points removed
    """
    
    Y = lex_sort(Y)
    
    Yn = PointList(())
    Yn.dim = Y.dim
    Yn.points = list(Yn.points)
    # phase 1
    for i, y in enumerate(Y):
        if not Yn.weakly_dominates_point(y):
            Yn.points.append(y)
    # print(f"Phase 1 result: {Yn=}")
    # phase 2
    Yn_new = PointList()
    Yn_new.dim = Yn.dim
    Yn_new.points = list(Yn_new.points)
    for i in range(len(Yn)):
        y = Yn[-i + 1]
        if not Yn_new.weakly_dominates_point(y):
            Yn_new.points.append(y)
    return Yn_new       


def KD_filter(Y: PointList):
    """ Kd-tree filtering algorithm from Chen2012

    Args:
        Y (PointList)

    Returns:
        Yn (PointList) set of nondominated points 

    Returns: PointList with all nondominated points removed

    """
    Y = lex_sort(Y) # sort input list
    R = []
    for i, y in enumerate(Y):
        if i == 0:
            r = KD_Node(Y[0], 0)
            r.UB = r.y
            r.LB = r.y
            k = 0
            R.append(y)
            continue

        if not KD_tree.dominates_point(r, y):
            # print(f"{y=}")
            KD_tree.insert(r,0, y)
            R.append(y)
            k += 1

    r = None
    Yn = []
    for i, y in enumerate(reversed(R)):
        if i == 0:
            r = KD_Node(y, 0)
            r.UB = r.y
            r.LB = r.y
            Yn.append(y)
            continue
        
        if not KD_tree.dominates_point(r, y):
            KD_tree.insert(r,0, y)
            Yn.append(y)

    Yn = PointList(Yn)

    return Yn


def lex_sort(Y: PointList):
    """
    input: PointList
    output: lexicographically sorted PointList Y

    source https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    """
    Y.points = sorted(Y.points, key=itemgetter(*range(Y.dim)))

    for i in range(len(Y.points)-1): # simple but not exhaustive correctness check
        assert not Y[i] > Y[i+1], f"{Y[i]=},{Y[i+1]=} "

    return PointList(Y.points)


def unidirectional_filter(Y: PointList, duplicates_allowed = False) -> PointList:
    """
    input: PointList, bool - allowed_duplicates
    output: PointList with all nondominated points removed
    """


    Y = lex_sort(Y)

    # p = 2
    assert Y[0].val.shape[0] <= 2, "dim p > 2 NOT IMPLEMENTED"
    Yn = []
    
    if duplicates_allowed:
        for y in Y:
            if Yn == [] or not Yn[-1] < y:
                Yn.append(y)
                # assert not PointList(Yn).dominates_point(y), f"{Yn=}, {y=}"
    
    else: # if duplicates not allowed
        for y in Y:
            if Yn == [] or not Yn[-1] <= y:
                Yn.append(y)
    return(PointList(Yn))

    # p > 2 NOT IMPLEMENTED



def call_c_nondomDC():
    current_d = os.getcwd()
    # move to c folder and execute
    os.chdir('/Users/au618299/Desktop/cythonTest/nondom/')
    subprocess.call(['./nondom'])
    # return to initial directory
    os.chdir(current_d)


def nondomDC_wrapper(Y : PointList):
    # A python wrapper for the c implementation of NonDomDC [Bruno Lang]
    out_file = fr"/Users/au618299/Desktop/cythonTest/nondom/pointsCin" # c script directory
    Y.save_raw(out_file)
    call_c_nondomDC()
    in_file = filepath = fr"/Users/au618299/Desktop/cythonTest/nondom/pointsCout" # c script directory
    return(PointList.from_raw(in_file))


def N(Y = PointList, **kwargs):
    """ 'best' implemented nondominance filter """
    if Y[0].dim <= 2:
        return unidirectional_filter(Y, *kwargs)
    # elif len(Y) < 1024*2:
        # return KD_filter(Y)
    else:
        return nondomDC_wrapper(Y)
        # return naive_filter(Y, MCtF = True)
        # return two_phase_filter(Y)

def MS_sum(Y_list = list[PointList], operator = "+") -> PointList:
    """
    input: list of PointList
    output: Minkowski sum of sets
    """
    assert operator in ("+","-","*")

    Y_ms = Y_list[0]
    for s in range(1, len(Y_list)):
        Y_ms_new = []
        Y_s = Y_list[s]
        for y_ms in Y_ms:
            for y_s in Y_s:
                if operator == "+":
                    Y_ms_new.append(y_ms+y_s)
                if operator == "-":
                    Y_ms_new.append(y_ms-y_s)
                if operator == "*":
                    Y_ms_new.append(y_ms*y_s)
        Y_ms = Y_ms_new

    return PointList(Y_ms)



def MS_naive_filter(Y_list = list[PointList]) -> PointList:
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """
    Y = MS_sum(Y_list)
    Yn = N(Y)

    return PointList(Yn)
 

def MS_sequential_filter(Y_list = list[PointList], N = N) -> PointList:
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """
    Y_ms = N(Y_list[0])

    for s in range(1, len(Y_list)):
        # print(f"{s=}")
        # print(f"{len(Y_ms)=}")
        Y_ms = N(Y_ms + N(Y_list[s]))

    return PointList(Y_ms)


def MS_doubling_filter(Y_list = list[PointList], MS_filter_alg = MS_sequential_filter) -> PointList:
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """

    s = len(Y_list)
    S = Y_list
    while s > 1:
        S_new = []
        for k in range(math.floor(s/2)):
            S_new.append(MS_filter_alg((S[2*k], S[2*k+1])))
        if s % 2 != 0:
            S_new.append(S[-1])
        s = math.ceil(s/2)
        S = S_new
    return S[0]
        
   



def lex_sort_linked(Y: PointList):
    """function for sorting p = 2$"""
    assert Y.dim <= 2, "dim p > 2 NOT IMPLEMENTED"
 
    llist = LinkedList()
    llist.add_first(Node(Y[0]))
    llist.head.prev = None

    for y_current in Y[1:]:
        for N in llist:
            if y_current.val[0] >= N.data.val[0]:
                continue
            # traverse linked list until y lex dominated the node N
            if y_current.lex_le(N.data):
                new_node = Node(y_current)
                # add before N
                llist.add_before(N.data, new_node)

                # remove N.data and all dominated children
                first_nondom = N
                # while first_nondom != None and y_current < first_nondom.data:
                    # first_nondom = first_nondom.next
                    # print(f"removing {first_nondom}")
                new_node.next = first_nondom
                break
           #  elif N.data < y_current:
           #      break
            prev = N
        else:
            if N.next == None:
                new_node = Node(y_current)
                N.next = new_node
    return PointList((N.data for N in llist.__iter__()))


def lex_filter(Y: PointList):
    """function for filtering out dominated points using linked lists for p = 2$"""
    assert Y.dim <= 2, "dim p > 2 NOT IMPLEMENTED"
 
    llist = LinkedList()
    llist.add_first(Node(Y[0]))
    llist.head.prev = None

    for y_current in Y[1:]:
        for N in llist:
            if y_current.val[0] >= N.data.val[0]:
                if y_current.val[1] >= N.data.val[1]:
                    break
                else:
                    continue
            # traverse linked list until y lex dominated the node N
            if y_current.lex_le(N.data):
                new_node = Node(y_current)
                # add before N
                llist.add_before(N.data, new_node)

                # remove N.data and all dominated children
                first_nondom = N
                while first_nondom != None and y_current < first_nondom.data:
                    first_nondom = first_nondom.next
                    # print(f"removing {first_nondom}")
                new_node.next = first_nondom
                break
           #  elif N.data < y_current:
           #      break
            prev = N
        else:
            if N.next == None:
                if not N.data < y_current:
                    new_node = Node(y_current)
                    N.next = new_node
    return PointList((N.data for N in llist.__iter__()))


def induced_UB(Y: PointList, line=False, assumption = "consecutive"):
    """ Induced upper bound set from pointlist Y, points are assumed to be consecutive in Yn"""
    # arg assumption in [consecutive, supported, nonconsecutive]
    assert assumption in ["consecutive", "supported", "nonconsecutive"]

    Y = N(Y)
    U = []
    seen = set() # for spotting duplicates
    if line:
        U.append(Y[0])
        for i in range(len(Y)-1):
                if assumption == "consecutive":
                    u = Point((Y[i+1][0], Y[i][1]))
                elif assumption == "nonconsecutive":
                    u = Point((Y[i][0], Y[i+1][1]))
                if assumption != "supported":
                    U.append(u)
                U.append(Y[i+1])
    else:
        for i in range(len(Y)-1):
            if Y[i+1] not in seen: # ignore duplicates
                seen.add(Y[i+1])
                u = Point((Y[i+1][0], Y[i][1]))
                U.append(u)
    U = PointList(U)
    return U
 



def find_generator_U(Y1:PointList, Y2:PointList) -> PointList:
    """
    input: two sets Y1, Y2, where Y1 contains (global) lex min solutions.
    output: A set of generator upper bound points Uc as PointList
    """

    def get_i(points: PointList, q: Point):
        """
        intervals: a PointList y1_1 < y2_1 < y3_1 ... 
        q: a Point
        returns the id i of PointList where yi_1 == q_1
        """
        if points[0][0] == q[0]:
            return 0
        if points[-1][0] <= q[0]:
            return -1 

        # assert that Q is sorted (consequence of Y2 sorted)
        for i, y in enumerate(points):
            if points[i][0] <= q[0] and q[0] < points[i+1][0]:
                return i

    Y1 = N(Y1)
    Y2 = N(Y2)

    Y = N(Y1+Y2)

    y_ul = Y1[0]
    y_lr = Y1[-1]

    u_current = y_ul
    Uc = [u_current]
    Q = PointList((u_current,)) + Y2

    while u_current != y_lr:
        #assert Q == PointList((u_current,)) + Y2
        # determine right movement
        Q_bar = [q for q in Q if Y[get_i(Y,q)][1] == q[1]]
        l1 = max([Y[get_i(Y,q)+1][0]-q[0] for q in Q_bar])

        # determine down movement
        Q = Point((l1,0)) + Q
        u_current = u_current + Point((l1, 0))
        l2 = min([q[1] - Y[get_i(Y,q)][1] for q in Q])

        # Update Q, u_current and Uc
        Q = Point((0,-l2)) + Q
        u_current = u_current + Point((0, -l2))
        Uc.append(u_current)

    return PointList(Uc)




