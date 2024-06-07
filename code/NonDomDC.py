from classes import PointList
from methods import unidirectional_filter, KD_filter, N, naive_filter
import math
import matplotlib.pyplot as plt


def union(*args):
    points = []
    for Y in args:
        points += Y.points
    return PointList(points).removed_duplicates()

# def union(Y1, Y2):
    # return PointList(Y1.points + Y2.points)


def split(Y : PointList, d : int, pivot = None):
    # determine M = median on the d'th coordinate
    Y = sorted(Y, key = lambda x: x[d-1], reverse=False)
   
    if not pivot:
        pivot = Y[math.floor(len(Y)/2)]
        # pivot_first_check = Y.index(pivot)
        # pivot_last_check = len(Y) - Y[::-1].index(pivot)

    print(f"Q split")
    pivot_first = 0
    for i,y in enumerate(Y):
        if y[d-1] < pivot[d-1]:
            pivot_first = i + 1
            # pivot_last = pivot_first
        if y[d-1] == pivot[d-1]:
            pass
        if y[d-1] > pivot[d-1]:
            pivot_last = i
            break

    # partition on pivot element lt, eq, gt
    Ylt = Y[:pivot_first]
    Yeq = Y[pivot_first: pivot_last]
    Ygt = Y[pivot_last:]

    print(f"{len(Y)=}")
    print(f"{len(Ylt)=}")
    print(f"{len(Yeq)=}")
    print(f"{len(Ygt)=}")
    assert len(Y) == len(Ylt) + len(Ygt) + len(Yeq)


    print(f"")
    print(f"{Y=}")
    del Y
    print(f"{d=}")
    print(f"{Ylt=}")
    print(f"{Yeq=}")
    print(f"{Ygt=}")

    # Ygt must have smaller d'th  (0 index => d-1) coordinate than Ylt
    print(f"{pivot=}")
    assert Ylt[0][d-1] < Ygt[0][d-1] 

    return PointList(Ylt), PointList(Yeq), PointList(Ygt), pivot


def filterDC(Y: PointList, Q : PointList, d: int):
    # Those points 𝐩 ∈ 𝐏 that are not (non-strictly) dominated by any point in Q wrt to the first d coordinates.
    if d<= 1:
        min_q_d = min([q[d] for q in Q])
        # return PointList([y for y in Y if y[0] < min_q_d])
        return PointList([y for y in Y if not any((q.lt_d(y,d) for q in Q))])

    elif len(Y)*len(Q) <= 1024:
        return PointList([y for y in Y if not any((q.lt_d(y,d) for q in Q))])



    # split sets on pivot element
    Ylt, Yeq, Ygt, pivot = split(Y,d)
    Qlt, Qeq, Qgt, _ = split(Q,d, pivot)

    Ylt_F = filterDC(Ylt, Qlt, d)
    Yeq_F = filterDC(Yeq, union(Qlt , Qeq), d - 1)
    Ygt_F = filterDC(filterDC(Ygt, union(Qlt , Qeq), d-1), Ygt, d)

    return PointList(Ylt_F.points + Yeq_F.points + Ylt_F.points)

def NonDomDC(Y : PointList, d : int):
    if len(Y) == 0:
        return Y
    if d <= 2:
        print(f"low dim")
        return KD_filter(Y)
    elif len(Y) <= 64:
        print(f"low nr points {len(Y)}")
        return KD_filter(Y)

    
    Ylt, Yeq, Ygt, _ = split(Y,d)

    Ylt_N = NonDomDC(Ylt, d) # Yly points can only be dominated by other points from Ylt
    Yeq_N = filterDC(NonDomDC(Yeq,d-1), Ylt_N, d-1) # Yeq can be dominated by points from Ylt_N and Yeq, only on the d-1 coordinate
    Ygt_N = filterDC(NonDomDC(Ygt,d), union(Ylt_N, Yeq_N), d - 1) # Ygt points can be dominated by points from itself (all d coordinates) and points in Ylt_N or Yeq_N if dominated on the d-1'th coordinate.
    
    return union(Ylt_N , Yeq_N , Ygt_N)
    
def test():

    Y1 = PointList.from_json("./instances/subproblems/sp-3-10-m_1.json")
    Y2 = PointList.from_json("./instances/subproblems/sp-3-10-m_2.json")
    Y = Y1 + Y2
    Yn = NonDomDC(Y, Y.dim)


    Yn_correct = naive_filter(Y)
    # print(f"{Yn=}")
    print(f"{len(Yn)=}")
    print(f"{len(Yn_correct)=}")


    fig = plt.figure()
    ax= plt.axes(projection = '3d')

    Y.plot("Y",ax=ax )
    Yn.plot("Yn",ax=ax)
    Yn_correct.plot("Yn*", ax=ax)
    plt.show()

    not_in_Yn_correct = [y for y in Yn if y not in Yn_correct]
    not_in_Yn = [y for y in Yn_correct if y not in Yn]

    print(f"{len(not_in_Yn)=}")
    print(f"{len(not_in_Yn_correct)=}")
    assert Yn == Yn_correct

def main():
    test()

if __name__ == "__main__":
    main()  
