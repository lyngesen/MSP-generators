"""
File containing code for plots used in tex project.

Usage:
    Run as main to update plots with new style.
    change style in if main statement
"""

import sys
sys.path.insert(0, "../../code/")

from classes import Point, PointList, LinkedList
import methods
from methods import N
import matplotlib.pyplot as plt
import matplotlib

def plot_or_save(fig, fname: str):
    """ call to plot or save fig """
    if SAVE_PLOTS:
        fig.savefig(fname = FIGURES_LOCATION + f"{fname}.pgf")
        fig.savefig(fname = FIGURES_LOCATION + f"{fname}.pdf")
    else:
        plt.title(fname)
        plt.show()

    
def example():
    N_list = range(10)

    Y1 = PointList(([(i,0) for i in N_list]))
    Y2 = PointList(([(i,i) for i in N_list]))
    Y3 = PointList(([(i,len(N_list)-1-i) for i in N_list]))

    # Y1.plot()
    # Y2.plot()
    # Y3.plot()
    Y1.plot()
    Y2.plot()
    Y3.plot()
    Y = methods.MS_sum((Y1,Y2))
    Y = PointList([y+Point((20,0)) for y in Y])
    Y.plot()
    Y = methods.MS_sum((Y1,Y2,Y3))
    Y = PointList([y+Point((50,0)) for y in Y])
    Y.plot()
    plt.show()


def klamroth2023_lemma2():
    l = 6
    m = 6
    A = PointList([(-1,-1,-0)] + [(-0, -i/l, -(l-i)/l) for i in range(1,l)])
    B = PointList([(-0,-1,-1)] + [(-i/m, -(m-i)/m, -0 ) for i in range(1,m)])

    print(f"{A=}")
    print(f"{B=}")


    fig = plt.figure()
    ax= plt.axes(projection = '3d')
    

    A.plot(ax= ax, l="A")
    B.plot(ax= ax, l="B")

    # S = PointList(A.points + B.points)
    S = A + B

    S.plot(ax= ax, l = "A + B")
    Sn = N(S)
    print(f"A+B = {S}")
    print(f"(A+B)_N = {Sn}")
    Sn.plot(ax= ax, l= "(A + B)_N" , color="blue")

    print(f"{len(Sn.removed_duplicates())=}")

    print(f"|A|={len(A)}")
    print(f"|B|={len(B)}")
    print(f"|A+B|={len(S)}")
    print(f"|(A+B)_N|={len(Sn)}")

    plt.show()


def main():

    klamroth2023_lemma2()
    # example()



if __name__ == '__main__':

    SAVE_PLOTS = False
    FIGURES_LOCATION = "figures/"
    NO_AXIS = False 

    # used figure sizes
    SIZE_STANDARD_FIGURE = (5,2)
    SIZE_SMALL_FIGURE = (2.5,2)
    SIZE_LARGE_FIGURE = (5,3)

    # Style options for plots
    # all_styles = ['Solarize_Light2', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',  'tableau-colorblind10']
    STYLE = "ggplot" 
    plt.style.use(STYLE)


    if NO_AXIS:
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['ytick.left'] = False
        plt.rcParams['ytick.labelleft'] = False

    if SAVE_PLOTS:
        matplotlib.rcParams.update({"pgf.texsystem": "pdflatex",})

    matplotlib.rcParams.update({
            'font.family': 'serif',
            'text.usetex': True,
            # 'font.size' : 11,
            'pgf.rcfonts': False,
            # 'grid.alpha': 0.0,
            # 'figure.figsize': (8.77, 4.2),
        })

    # latex commands
    _Y = "\mathcal{Y}" # % objective space
    _A = "\mathcal{A}" # % a set
    _B = "\mathcal{B}" # % a set
    _U = "\mathcal{U}" # % upper bound set _Yn = "\mathcal{Y}_\mathcal{N}" # % non-dominated objective space
    _Y1 = "\mathcal{Y}^1" # % objective space 1
    _Y2 = "\mathcal{Y}^2" # % objective space 1
    _Yn = "\mathcal{Y}_\mathcal{N}" # % non-dominated objective space
    _Yn1 = "\mathcal{Y}_\mathcal{N}^1" # % non-dominated objective space
    _Yn2 = "\mathcal{Y}_\mathcal{N}^2" # % non-dominated objective space
    _mcN = "\mathcal{N}" # % Non-dominated set
    _YsS = "\mathcal{Y}^{s|S}" # % Conditioned non-dominated set
    _Gc = "\mathcal{G}^{1|2}" # % set of generators
    _Yc = "\\bar{\mathcal{Y}}^{1}" # % a minimal generator
    _hYc = "\hat{\mathcal{Y}}^{1}" # % (another) minimal generator
    _hYn = "\hat{\mathcal{Y}}_\mathcal{N}" # % subset of _Yn
    _Uc = "\\bar{\mathcal{U}}^{1}" # % a conditional upper bound set
    _bY = "\\bar{\mathcal{Y}}" # % bar over objective space
    _bYn = "\\bar{\mathcal{Y}}_\mathcal{N}" # % bar over non-dominated points

    main()
