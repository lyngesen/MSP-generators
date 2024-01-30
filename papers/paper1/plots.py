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


def UB_types():

    ######################## Figure consequtive START ########################
    fig_names = ["consecutive", "nonconsecutive", "supported", "mixed"]
    # define new figure
    
    Y = PointList(((1,10), (4, 6), (7,3), (10,2)))
    U_mixed = PointList(((1,10), (4,6),(7,6), (7,3), (7,2),(10,2)))

    for fig_name in fig_names:
        print(f"Plotting figure: {fig_name}")
        fig, ax = plt.subplots(figsize=SIZE_SMALL_FIGURE, layout='constrained')

        if fig_name in ["consecutive", "nonconsecutive", "supported"]:
            U = methods.induced_UB(Y, line = True, assumption =fig_name)
        elif fig_name == "mixed":
            U = U_mixed
        Y.plot(l=f"${_hYn}$", point_labels=True)
        U.plot(l=f"${_U}$", line=True, linestyle = "dashed")
        # save or plot figure
        plot_or_save(fig, "ub_types_" + fig_name)
    ######################### Figure consequtive END #########################



def main():

    UB_types()



if __name__ == '__main__':

    SAVE_PLOTS = True
    FIGURES_LOCATION = "figures/"
    NO_AXIS = True 

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
