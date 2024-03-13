"""
File containing code for plots used in tex project.

Usage:
    Run as main to update plots with new style.
    change style in if main statement
"""

import sys
import numpy as np
sys.path.insert(0, "../../code/")

from classes import Point, PointList, LinkedList, MinkowskiSumProblem
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

def plot_ul():

    ######################## Figure plot_ul START ########################
    fig_name = "plot_ul"
    print(f"Plotting figure: {fig_name}")
    # define new figure
    fig, ax = plt.subplots(figsize=SIZE_STANDARD_FIGURE, layout='constrained')
    
    MSP = MinkowskiSumProblem.from_json('instances/problems/prob-2-100|100-ul-2_1.json')
    Y1, Y2 = MSP.Y_list

    Y1.label = f"${_Yn1}$"
    Y2.label = f"${_Yn2}$"

    for Y in MSP.Y_list:
        Y.plot(l = Y.label)
    

    
    # save or plot figure
    plot_or_save(fig, fig_name)
    ######################### Figure plot_ul END #########################


def matrix_plot(Y1,Y2, fig_name, point_labels = False, matrix_only = False):
    """ plots similar to matrix plots in Hespe et al. 2023 """
    
    Y1 = methods.lex_sort(Y1)
    Y2 = methods.lex_sort(Y2)
    Y1_color = Y1.plot_color   
    Y2_color = Y2.plot_color   

    Y = Y1+Y2
    Yn = N(Y)
    Yn_set = set(Yn.points)

    # ######################## Figure ax1 START ########################
    # fig_name = "ax1"
    # # define new figure
    

    fig = plt.figure(figsize=SIZE_VERY_LARGE_FIGURE, layout='constrained')
    ax = list()
    projection = None if Y1.dim == 2 else '3d'
    ax.append(fig.add_subplot(1, 3, 1, projection= projection ))
    ax.append(fig.add_subplot(1, 3, 2))
    ax.append(fig.add_subplot(1, 3, 3, projection= projection ))
    # fig, ax = plt.subplots(ncols = 3, figsize=SIZE_LARGE_FIGURE, layout='constrained')
    
    if not matrix_only:
        Y1.plot(ax = ax[0], l = f"${_Yn1}$", point_labels= point_labels)
        Y2.plot(ax = ax[0], l = f"${_Yn2}$", point_labels= point_labels)

    # # save or plot figure
    # plot_or_save(fig, fig_name)

    # add marking of point contributes to Pareto Sum point:

    Y1_MSP = PointList([y1 for y1 in Y1 if any((y1+y2 in Yn_set for y2 in Y2))])
    Y2_MSP = PointList([y2 for y2 in Y2 if any((y1+y2 in Yn_set for y1 in Y1))])
    if not matrix_only:
        Y1_MSP.plot(ax = ax[0], l = f"${_Yn1}\\rightarrow {_Yn}$",color = "yellow", marker='1')
        Y2_MSP.plot(ax = ax[0], l = f"${_Yn2}\\rightarrow {_Yn}$", color = "yellow", marker='1')

    # ######################### Figure ax1 END #########################



    M = np.array([[1 if y1+y2 in Yn_set else 0 for y1 in Y1] for y2 in Y2])
    col_header = np.array([3 for _,y  in enumerate(Y1)])
    row_header = np.array([5] + [4 for _,y  in enumerate(Y2)])

    M = np.vstack([col_header, M])
    M = np.vstack([row_header, M.transpose()])

    ######################## Figure matrix_plot START ########################
    
    # Define your custom colors
    colors = ['lightgray', 'yellow', 'red', Y1.plot_color, Y2.plot_color, 'white']
    if matrix_only:
        colors = ['lightgray', 'yellow', 'red', 'red', 'blue', 'white']
    # Create a colormap with discrete colors
    cmap = matplotlib.colors.ListedColormap(colors)

#     ax[1].imshow(M, cmap = cmap)
    # # ax[1].imshow(M)

    if point_labels:
        for i, y in enumerate(Y1):
            ax[1].annotate(text = "$y^{" +  f"{i+1}" + "}$", xy = (i+1.5,0.51))   
        for i, y in enumerate(Y2):
            ax[1].annotate(text = "$y^{" +  f"{i+1}" + "}$", xy = (0.5,i+1.5))   

    ax[1].axis('off')


    ax[1].pcolormesh(M, cmap = cmap, edgecolors='white', linewidth=0.5)
    ax[1] = plt.gca()
    ax[1].set_aspect('equal')
    # ax[0].set_aspect('equal')
    # ax[2].set_aspect('equal')
    
    ax[0].set_title("Subproblem Pareto sets")
    ax[1].set_title("Matrix")
    
    print(f"|Yn| = {len(Yn)}")
    print(f"|Y1| + |Y2| = {len(Y1) + len(Y2)}")
    print(f"|Y1||Y2| = {len(Y1)*len(Y2)}")
    print(f"")
    print(f"|Y1| = {len(Y1)}")
    print(f"|Y2| = {len(Y2)}")
    print(f"|Yn1 MSP | = {len(Y1_MSP)}")
    print(f"|Yn2 MSP | = {len(Y2_MSP)}")
    
    # save or plot figure
    # plot_or_save(fig, fig_name)
    ######################### Figure matrix_plot END #########################


    ######################## Figure pareto_sum START ########################
    # fig_name = "pareto_sum"
    print(f"Plotting figure: {fig_name}")
    # define new figure
    # fig, ax = plt.subplots(figsize=SIZE_STANDARD_FIGURE, layout='constrained')
 
    if not matrix_only:
        Y.plot(ax = ax[2], l = f"${_Y}$", color = "lightgray")
        Yn.plot(ax = ax[2], l = f"${_Yn}$", color = "yellow")
    
    # save or plot figure
    plot_or_save(fig, fig_name)
    ######################### Figure pareto_sum END #########################
        
def make_matrix_plot():

    plot_configs = [
            {"fig_name": 'matrix_plot', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-10-l_1.json",
             'Y2' : "instances/subproblems/sp-2-10-u_1.json"},
            {"fig_name": 'methods_lu', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-u_1.json"},
            {"fig_name": 'methods_lm', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-m_1.json"},
            {"fig_name": 'methods_mm', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-100-m_1.json",
             'Y2' : "instances/subproblems/sp-2-100-m_2.json"},
            {"fig_name": 'scaling_lu_1', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-u_1.json"},
            {"fig_name": 'scaling_lu_2', 'scaling' : 2,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-u_1.json"},
            {"fig_name": 'scaling_lu_0_5', 'scaling' : 0.5,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-u_1.json"},
            {"fig_name": 'scaling_lu_0_1', 'scaling' : 0.1,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-u_1.json"},
            {"fig_name": 'dim_2', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-100-l_1.json",
             'Y2' : "instances/subproblems/sp-2-100-u_1.json"},
            {"fig_name": 'dim_3', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-3-100-l_1.json",
             'Y2' : "instances/subproblems/sp-3-100-u_1.json"},
            {"fig_name": 'dim_4', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-4-100-l_1.json",
             'Y2' : "instances/subproblems/sp-4-100-u_1.json"},
            {"fig_name": 'dim_5', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-5-100-l_1.json",
             'Y2' : "instances/subproblems/sp-5-100-u_1.json"},
            {"fig_name": 'magnitudes1', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-10-u_1.json",
             'Y2' : "instances/subproblems/sp-2-10-m_10.json"},
            {"fig_name": 'magnitudes10', 'scaling' : 10,
             'Y1' : "instances/subproblems/sp-2-10-u_1.json",
             'Y2' : "instances/subproblems/sp-2-10-m_10.json"},
            {"fig_name": 'magnitudes20', 'scaling' : 20,
             'Y1' : "instances/subproblems/sp-2-10-u_1.json",
             'Y2' : "instances/subproblems/sp-2-10-m_10.json"},
            {"fig_name": 'magnitudes1_1', 'scaling' : 1,
             'Y1' : "instances/subproblems/sp-2-10-u_1.json",
             'Y2' : "instances/subproblems/sp-2-10-m_10.json"},
            {"fig_name": 'magnitudes1_2', 'scaling' : -2,
             'Y1' : "instances/subproblems/sp-2-10-u_1.json",
             'Y2' : "instances/subproblems/sp-2-10-m_10.json"},
            {"fig_name": 'magnitudes1_5', 'scaling' : -5,
             'Y1' : "instances/subproblems/sp-2-10-u_1.json",
             'Y2' : "instances/subproblems/sp-2-10-m_10.json"},
            ]
    
    for P in plot_configs:
        Y1 = PointList.from_json(P['Y1'])
        Y2 = PointList.from_json(P['Y2'])
        fig_name = P['fig_name']
        scaling = P['scaling']

        # scaling = 1
        if scaling:
            if scaling < 0:
                Y1 = PointList([y*(-scaling) for y in Y1])
            else:
                Y2 = PointList([y*scaling for y in Y2])

        point_labels = (len(Y1) <= 10 and len(Y2) <= 10)

        matrix_only = Y1.dim > 3
        
        # fig_name = "matrix_plot"
        if 'magnitudes' not in fig_name: continue

        matrix_plot(Y1,Y2, fig_name = fig_name, point_labels= point_labels, matrix_only=matrix_only)


def test_matrix_plot():
    Y1 = PointList.from_json("instances/subproblems/sp-2-10-u_1.json")
    Y2 = PointList.from_json("instances/subproblems/sp-2-10-m_10.json")
    fig_name = "test_plot" 
    scaling = 10

    # scaling = 1
    if scaling:
        Y2 = PointList([y*scaling for y in Y2])

    point_labels = (len(Y1) <= 10 and len(Y2) <= 10)

    matrix_only = Y1.dim > 3
    
    # fig_name = "matrix_plot"

    matrix_plot(Y1,Y2, fig_name = fig_name, point_labels= point_labels, matrix_only=matrix_only)




def klamroth2023_lemma2():
    l = 6
    m = 6
    A = PointList([(-1,-1,-0)] + [(-0, -i/l, -(l-i)/l) for i in range(1,l)])
    B = PointList([(-0,-1,-1)] + [(-i/m, -(m-i)/m, -0 ) for i in range(1,m)])

    # print(f"{A=}")
    # print(f"{B=}")


    fig = plt.figure()
    fig.tight_layout(h_pad=20)
    ax= plt.axes(projection = '3d')
    

    A.plot(ax= ax, l=f"${_A}$")
    B.plot(ax= ax, l=f"${_B}$")

    # S = PointList(A.points + B.points)
    S = A + B

    S.plot(ax= ax, l = f"${_A} + {_B}$")
    Sn = N(S)
    # print(f"$A+B = \{S}$")
    # print(f"(A+B)_N = {Sn}")
    Sn.plot(ax= ax, l= f"(${_A} + {_B} )_{_mcN}$" , color="blue")

    # print(f"{len(Sn.removed_duplicates())=}")

    # print(f"|A|={len(A)}")
    # print(f"|B|={len(B)}")
    # print(f"|A+B|={len(S)}")
    # print(f"|(A+B)_N|={len(Sn)}")

    plt.show()



def main():

    # plot_ul()

    # klamroth2023_lemma2()
    # example()

    # with plt.style.context('bmh'):

    # with plt.rcParams({"lines.linewidth": 2, "lines.color": "r"}):
    # with matplotlib.rc_context['xtick.bottom']:

    # test_matrix_plot()
    make_matrix_plot()

if __name__ == '__main__':

    SAVE_PLOTS = True 
    FIGURES_LOCATION = "../../papers/paper1/figures/matrix_plots/"
    NO_AXIS = False 

    # used figure sizes
    SIZE_STANDARD_FIGURE = (5,2)
    SIZE_SMALL_FIGURE = (2.5,2)
    SIZE_LARGE_FIGURE = (5,3)
    SIZE_VERY_LARGE_FIGURE = (10,3)

    # Style options for plots
    # all_styles = ['Solarize_Light2', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',  'tableau-colorblind10']
    STYLE = "ggplot" 
    plt.style.use(STYLE)


    if NO_AXIS:
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['ytick.left'] = False
        plt.rcParams['ytick.labelleft'] = False

    # if SAVE_PLOTS:
        # matplotlib.rcParams.update({"pgf.texsystem": "pdflatex",})

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
