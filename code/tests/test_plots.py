from classes import Point, PointList, LinkedList
import methods
import matplotlib.pyplot as plt
import matplotlib

def test_point():
    y = Point((1,3))
    y.plot()


def test_pointlist():
    Y = PointList.from_csv("instances/testsets/BINOM-p2-n10-s1")
    Y.plot()


def test_3d_plots():

    Y1 = PointList.from_csv("instances/testsets/CONCAVE-p3-n10-s1")
    Y2 = PointList.from_csv("instances/testsets/DISK-p3-n10-s1")
    Y = Y1 + Y2
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    Y1.plot(ax = ax, l ="")
    Y2.plot(ax = ax, l ="")
    # Y.plot(ax = ax, l ="$\mathcal{Y}$")
    Yn = methods.naive_filter(Y)
    Yn.plot(ax = ax, l ="")
    print(f"{len(Y)=}")
    y = Point((1,1,1))
    y.plot(ax = ax, l="$y$")



    y3d = Point((1,2,3))
    y3d.plot(ax=ax, l ="1")
    y3d.plot(ax=ax,l ="1", label_only=True)


