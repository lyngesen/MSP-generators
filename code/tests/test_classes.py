from classes import Point, PointList, MinkowskiSumProblem
from classes import Node, LinkedList 
import methods
import os

import pytest

def test_node():
    a = Node("A")
    a_str = a.__repr__()
    b = Node("B")
    a.next = b
    b.prev = a
    c = Node("C")
    b.next = c
    c.prev = b
    assert a.next.next == c
    assert a.next == b
    assert a == c.prev.prev

def test_linkedlist():
    L = LinkedList()

    a = Node("A")
    b = Node("B")
    L.add_first(a)
    print(f"{L}")
    L.add_after("A", b)


    assert a.next == b


    c = Node("C")
    L.add_before("B", c)

    assert a.next == c

    d = Node("D")
    L.add_first(d)
    L.remove_node("D")
    print(f"{L}")
    L.head == c



def test_points():
    y1 = Point((1,2))
    y2 = Point((3,2))

    assert y1[0] == 1
    assert y1[-1] == 2
    assert y1.dim == 2
    assert Point((1,2,3)).dim == 3
    print(f"{y1,y2}")

    assert y1 != y2
    assert y1 < y2
    assert y1 < y1 + y2
    assert y1 > y1 - y2
    assert not y2.lex_le(y1)
    assert y1[0] == 1
    assert not Point((1,2)).lex_le(Point((1,1)))

    # mul
    assert y1*y2 == Point((1*3, 2*2))
    assert y1*3 == Point((1*3, 2*3))
    assert y1*1.5 == Point((1*1.5, 2*1.5))
    with pytest.raises(TypeError):
        assert y1*"x"
    
    hash(y1)
    L = list(y1)
    y1.plot(l ="1")
    y1.plot(l ="1", label_only=True)


def test_pointlist():
    # read from points
    y1 = Point((1,2))
    y2 = Point((3,2))
    Y = PointList((y1,y2))
    assert y1 in Y
    assert Y.dominates_point(y2)
    assert not Y.dominates_point(y1)
    assert len(Y) == 2
    assert Y[1] == y2
    for y in Y:
        assert y in Y

    Y1 = PointList((y2, y1, y1))
    Y2 = PointList((y1, y1, y2))
    assert Y1 == Y2


    assert isinstance(PointList(y1)[0], Point) 
    assert isinstance(PointList(((1,1),))[0], Point) 

    # read from file
    testset = "instances/testsets/BINOM-p2-n100-s1"
    Y = PointList.from_csv(testset)

    Y1 = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')
    Y2 = PointList.from_json('instances/subproblems/sp-2-10-l_1.json')*2

    assert Y1 < Y2
    assert Y1 == Y1
    assert not (Y2 < Y1)
 
def test_pointlist_operators():
    y1 = Point((1,2))
    y2 = Point((3,2))
    # operations +, -, -
    Y1 = PointList((y1,y2))
    Y2 = PointList((y2))
    assert Y1 + Y2 == PointList((y1+y2, y2 + y2))
    assert Y1 + Y2 == methods.MS_sum((Y1,Y2), operator="+")
    assert Y1 - Y2 == methods.MS_sum((Y1,Y2), operator="-")
    assert methods.MS_sum((Y1,Y2), operator="-") == PointList((y1 - y2, y2 - y2))
    assert methods.MS_sum((Y1,Y2), operator="*") == PointList((y1 * y2, y2 * y2))


    # dominates
    U = PointList(((10,2),(5,5),(2,10)))
    L = PointList(((9,1),(4,5),(1,9)))
    assert L.dominates(U)
    assert L.dominates(U, power = 'strict')
    assert not U.dominates(U, power = 'strict')
    assert not U.dominates(U)
    assert not U.dominates(L)

    # others
    assert U == U.removed_duplicates()
    U2 = PointList(((10,2), (5,5), (2,10), (2,10)))
    print(f"{U}")
    print(f"{U2}")
    assert U == U2.removed_duplicates()


def test_PointList_json():
    # test small
    jsonfilename = "instances/subproblems/sp-2-10-l_1.json"
    print(f"{jsonfilename=}")
    Y = PointList.from_json(jsonfilename)
    Y.save_json("tests/temp/PointList_small_save_json.json")
    # test large
    jsonfilename = "instances/results/algorithm1/alg1-prob-4-50|50|50|50-uuuu-4_2.json"
    print(f"{jsonfilename=}")
    Y = PointList.from_json(jsonfilename)
    save_filename = "tests/temp/PointList_large_save_json.json"
    Y.save_json("tests/temp/PointList_large_save_json.json")
    os.remove(save_filename)



def test_MSP_json():
    jsonfilename = "instances/problems/prob-2-200|200-ll-2_1.json"
    MSP = MinkowskiSumProblem.from_json(jsonfilename)
    print(f"{MSP=}")



