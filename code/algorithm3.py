"""
File containing code for algorithm3 answering empirical research question 3

Usage:
    Run to add computational results
    output saved in: ./instances/results/algorithm3/result.csv
"""

from classes import Point, PointList, LinkedList, MinkowskiSumProblem
import methods
from methods import N
import time
import csv
import math
import os
from minimum_generator import solve_MGS_instance

import numpy as np
# from numoy import linalg
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from functools import reduce
import itertools
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def induced_UB_plot(level, Y1,Y2, prefix='', plot=True):
    print(f"{prefix}")
    # print(f"{level=}")
    def get_partial(Y, level='all'):   
        Y = N(Y)
        Y2e_points = [y for y in Y if y.cls == 'se']
        Y2other_points = [y for y in Y if y.cls != 'se']
        # random.shuffle(Y2other_points)
        match level:
            case 'all':
                return Y
            case 'lexmin': 
                return PointList((Y[0], Y[-1]))
            case 'extreme':
                return PointList(Y2e_points)
            # case float():
            case _:
                to_index = math.floor(float(level)*len(Y2other_points))
                return PointList(Y2e_points + Y2other_points[:to_index])
                # print(f"case not implemented {level}")
    
    
    Y2_partial = get_partial(Y2, level)


    ub_time = time.time()
    U = methods.find_generator_U(Y2_partial, Y1)
    ub_time = time.time() - ub_time

    Uline = methods.induced_UB(U,line=True)
   
    Y2_dominated = [y for y in Y2 if y.cls != 'se' and U.dominates_point(y)]
    dominated_relative = len(Y2_dominated)/len(Y2)
    print(f"dominated: {len(Y2_dominated)} \nrelative: {dominated_relative*100}\%")
    
    run_data = {'prefix' : prefix,
                'Y1_size' : len(Y1),
                'Y2_size' : len(Y2),
                'U' : len(U),
                'U_time' : ub_time,
                'dominated_points' : len(Y2_dominated),
                'dominated_relative_Y2' : dominated_relative,
                }

    return run_data
def multiple_induced_UB():


    set_options = ['l','m','u']
    size_options = [10, 50, 100, 150, 200, 300, 600]
    seed_options = [1,2,3,4,5]
    # UB_options = ['lexmin','extreme','0.25','0.5','0.75','all']
    UB_options = ['extreme']

    csv_file_path = './instances/results/algorithm3/result_slides_alg1_2.csv'
    # get last row
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            lastrow = row
    start_runs = False

    for s1 in size_options:
        s2 = s1
        for ub_level in UB_options:
        # s1 = 100 
        # s2 = 100
            # for s2 in size_options:
            for t1 in set_options:
                for t2 in set_options:
                    for seed in seed_options:
                        prefix = f'{t1}-{t2}_{s1}_{s2}_{ub_level}_{seed}_'
#                         if start_runs == False:
                            # if prefix == lastrow[0]:
                                # start_runs = True
                                # print(f"Starting run after {prefix}")
                            # continue
                        Y1 = PointList.from_json(f"./instances/subproblems/sp-2-{s1}-{t1}_{seed}.json")
                        Y2 = PointList.from_json(f"./instances/subproblems/sp-2-{s2}-{t2}_{max(seed_options)+1-seed}.json")
                        data = induced_UB_plot(ub_level, Y1,Y2, prefix, plot=False) 
                        data.update({'t1':t1, 't2':t2, 's1':s1, 's2':s2,'seed':seed,'ub_level':ub_level})
                        print(f"ALG1 solving Yn")
                        Y = Y1+Y2
                        Yn = N(Y)
                        data.update({'Y_size':len(Y), 'Yn_size':len(Yn)})
                        print(f"Solving MSG")
                        G = solve_MGS_instance([Y1,Y2])
                        data.update({'G1_size':len(G[0]), 'G2_size':len(G[1])})
                        with open(csv_file_path, 'a') as csv_file:
                            # add header if file empty
                            writer = csv.writer(csv_file)
                            if os.path.getsize(csv_file_path) == 0:
                                writer.writerow(data.keys())
                            writer.writerow(data.values())

def induced_LB_3d(Y : PointList, level: int, PLOT = False):

    cnames = {
    # 'aliceblue':            '#F0F8FF',
    # 'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    # 'azure':                '#F0FFFF',
    # 'beige':                '#F5F5DC',
    # 'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    }

    # for colorname, i in enumerate(cnames.keys()):
        # print(f"{colorname,i=}")



    cnames_list = list(cnames.values())
    for _ in range(10):
        cnames_list += cnames_list
    

    def in_hull(Y : PointList, y:Point):
        ''' from https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl '''
        points = Y.as_np_array()
        x = y.val
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        # print(lp.values())
        if not lp.success:
            return False
        else:
            return sum(1 for x in lp.x if x != 0)
        # return lp.success
    def strict_in_hull(Y : PointList, y:Point):

        if Y.dim ==2:
            return False 
        points = Y.as_np_array()
        epsilon = 1
        x = y.val

        centroid = np.mean(points, axis=0)
        # move all extreme points epsilon distance towards centroid
        for y in points:
            d = centroid - y
            # print(f"{d=}")
            # d = d / np.linalg.norm(d)
            y += d*epsilon

        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        # print(lp.values())
        if not lp.success:
            return False
        else:
            return sum(1 for x in lp.x if x != 0)
        # return lp.success





    def plot_surface(Y,ax,color = 'blue'):

        vertices = [tuple(y.val) for y in Y]
        if Y.dim == 2:
            PointList(vertices).plot(ax=ax, line=True, color=color, alpha = 0.2)
        else:
            ax.add_collection3d(Poly3DCollection([vertices], color=color, alpha = 0.2))

    def hull_sort(F: PointList):
        print(f"{len(F)=}")
        hull = ConvexHull(np.array([y.val for y in all_points]))
        FACES = tuple(PointList([y for i,y in enumerate(F) if i in sim]) for sim in hull.simplices)
        print(f"sdsad {len(FACES)=}")
        # assert len(FACES) == 1
        return FACES[0]

    def sort_polygon_vertices(Y:PointList):
        # Calculate the centroid of the polygon
        # centroid = y_bar.val

        vertices = np.array([y.val for y in Y])
        # print(f"{vertices=}")
        centroid = np.mean(vertices, axis=0)
        # centroid = reduce(Point.__add__, Y) * (1/len(Y))
        # Calculate the angles between each vertex and the centroid
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        
        # Sort the vertices based on the angles
        sorted_indices = np.argsort(angles)
        sorted_vertices = vertices[sorted_indices]
        return PointList(sorted_vertices)

    def plot_dominated_cone(point):
        pass

    def split_faces(FACES) -> list[PointList]:

        F_splits = []
        for f, F in enumerate(FACES):
            y_bar = reduce(Point.__add__, F) * (1/len(F))
            for i1,y1 in enumerate(F):
                    # F_noy1 = PointList([y for y in F if y != 0])
                    F_noy1 = PointList([y for y in F ])
                    # edge_points = [ye := (y1+y2)*(1/2) for i2,y2 in enumerate(F) if (y1 != y2 and not strict_in_hull(F_noy1,ye))]
                    edge_points = [(y1+y2)*(1/2) for i2,y2 in enumerate(F) if (y1 != y2)]
                    edge_points_outer = []
                    for ye in edge_points:
                        if not strict_in_hull(F, ye):
                            edge_points_outer.append(ye)
                    edge_points = edge_points_outer

                    for ye in edge_points:
                        # print(f"{F}")
                        assert in_hull(F, ye)
                    # new_face = PointList([y1]  + [y_bar]+ [y for y in edge_points])
                    new_face = PointList([y1]  + [y_bar]+ [y for y in edge_points])
                    new_face = sort_polygon_vertices(new_face)
                    # new_face.plot(ax=ax, l='{p}')
                    # plot_surface(new_face, ax, color = cnames_list[i1])
                    if len(new_face)>Y.dim-1:
                    # if True:
                        F_splits.append(new_face)

        return F_splits

    # fig = plt.figure()
    # ax= plt.axes(projection = '3d')
    all_points = PointList(list(Y.points) + [Y.get_nadir()])
    hull = ConvexHull(np.array([y.val for y in all_points]))
    # hull = ConvexHull(all_points.as_np_array())
    # Y = [PointList([y for i,y in enumerate(Y) if i in F]) for F in hull.simplices][4]
    # plot_surface(Y,ax)

    ZERO = Point([0 for _ in range(Y.dim)])
    nadir_point = Y.get_nadir()
    # nadir_point.plot(ax=ax,l=f"$y^N$")
    # ZERO.plot(ax=ax,l=f"$0$")
    if False: # plot axis lines
        for p in range(nadir_point.dim):
            unit_point = Point([nadir_point[q] if q ==p else 0 for q in range(nadir_point.dim)])
            unit_point.plot(ax=ax,l=f"$obj^{p}$", color='black')
            PointList([ZERO, unit_point]).plot(ax=ax, line=True, color='black')


    all_points = PointList(list(Y.points) + [nadir_point])
    hull = ConvexHull(np.array([y.val for y in all_points]))
    # hull = ConvexHull(np.array([y.val for y in Y]))
    # print(f"{hull.simplices=}")



    # FACES = tuple(hull.simplices)
    FACES = [PointList([y for i,y in enumerate(Y) if i in F]) for F in hull.simplices]

    if False:
        F_splits = []
        for f, F in enumerate(FACES):
            # if len(Y) in sim: #skip faces with nadir_point
                # print(f"skipping {sim}")
                # continue
            # y_bar = Point(sum([y for y in surfacePoints])/len(surfacePoints))

            # if f == 14:
            if True:
                # plot_surface(F, ax, color = cnames_list[f])
                y_bar = reduce(Point.__add__, F) * (1/len(F))
                y_bar.plot(ax=ax, l=r"$\bar{y}^{" + str(f) + "}$", color = 'black')
                l = F.get_ideal()
                u = F.get_nadir()
                l.plot(ax=ax, color = cnames_list[f])

                # F.plot(ax=ax,point_labels=True)
                # project y_bar onto each axis
                if True:
                    for i1,y1 in enumerate(F):
                        edge_points = [(y1+y2)*(1/2) for i2,y2 in enumerate(F) if y1 != y2]
                        new_face = PointList([y1]  + [y_bar]+ [y for y in edge_points])
                        new_face = sort_polygon_vertices(new_face)
                        new_face.plot(ax=ax, l='{p}')
                        plot_surface(new_face, ax, color = cnames_list[i1])
                        F_splits.append(new_face)
                        
                if False: # plot axis lines
                    edge_points = [(y1+y2)*(1/2) for (i1,y1) in enumerate(F) for i2,y2 in enumerate(F) if i1 < i2]
                    PointList(edge_points).plot(ax=ax,l=f"e", color = 'yellow')
                    for p in range(nadir_point.dim):
                        if p != 1:
                            pass
                            # continue
                        unit_point = Point([nadir_point[q] if q ==p else 0 for q in range(nadir_point.dim)])
                        projection = Point([y_bar[q] if q ==p else l[q] for q in range(nadir_point.dim)])
                        unit_point.plot(ax=ax,l=f"$obj^{p}$", color='black')
                        # projection = l + avg_direction
                        projection.plot(ax=ax,l=f'{p}')
                        PointList([l, unit_point + l]).plot(ax=ax, line=True, color='black')
                
                        # new_face = PointList([y for y in F if l[p] != y[p]] + [y_bar])
                        new_face = PointList([y for y in F if projection < y]  + [y_bar]+ [y for y in edge_points if projection < y])
                        # new_face = methods.lex_sort(new_face)
                        new_face = sort_polygon_vertices(new_face)
                        new_face.plot(ax=ax, l='{p}')
                        print(f"{len(new_face)=}")
                        print(f"{new_face.points}")
                        plot_surface(new_face, ax, color = cnames_list[f+p])
                        F_splits.append(new_face)

    # plt.show()


    for _ in range(level):
        print(f"{len(FACES)=}")


        if True:

            if PLOT:
                fig = plt.figure()
                if Y.dim == 3:
                    ax= plt.axes(projection = '3d' if Y.dim ==3 else '2d')
                    # ax.view_init(elev=-10., azim=200)
                    # ax.dist = 5
                else:
                    ax= plt.axes()
                Y.plot(ax = ax, l=r"$\mathcal{Y}$", color='red')
                # print(f"{F_splits=}")
            for f, F in enumerate(FACES):
                l = F.get_ideal()
                assert all((l <= y for y in F))
                if PLOT and len(F)>2:
                    # F.plot(ax=ax,point_labels=False)
                    l.plot(ax=ax, color = cnames_list[f])
                    plot_surface(F, ax, color = cnames_list[f])


        FACES = split_faces(FACES)

        # if _ == 0:
            # FACES = [FACES[1]]

    if PLOT: 
        plt.show()


    # return PointList(itertools.chain.from_iterable(((f.ideal for f in F) for F in FACES)))
    L = PointList((f.get_ideal() for f in FACES))
    return L

def algorithm3_pair(L_Y_U: list[list[PointList]]) -> list[PointList]:
    ((L1, Y1, U1), (L2, Y2, U2)) = L_Y_U
    U = methods.N(U1 + U2)
    

    Y1_dom = set()
    for y1 in Y1:


        if U < y1 + L2:
            print(f"{y1=} is dominated")
            Y1_dom.add(y1)
    G1 = PointList((y1 for y1 in Y1 if y1 not in Y1_dom))
    # repeat
    Y2_dom = set()
    for y2 in Y2:

        if U < y2 + L1:
            print(f"{y2=} is dominated")
            Y2_dom.add(y2)
    G2 = PointList((y2 for y2 in Y2 if y2 not in Y2_dom))
 
    print(f"{len(G1),len(Y1)=}")
    print(f"{len(G2),len(Y2)=}")
    

    if True:
        fig = plt.figure()
        if Y1.dim == 3:
            ax= plt.axes(projection = '3d')
        else:
            ax= plt.axes()
        print(f"{U < y2 + L1=}")
        L1.plot(ax=ax,l=f"L^1")
        L2.plot(ax=ax,l=f"L^2")
        U1.plot(ax=ax,l=f"U^1")
        U2.plot(ax=ax,l=f"U^2")
        U.plot(ax=ax,l=f"U")
        Y1.plot(ax=ax,l=f"Y^1")
        Y2.plot(ax=ax,l=f"Y^2")
        if Y2_dom:
            PointList(Y2_dom).plot(ax=ax, marker='x', color='black')
        if Y1_dom:
            PointList(Y1_dom).plot(ax=ax, marker='x', color='black')
        y2.plot(ax=ax,l='y2', label_only=True)
        (y2 + L1).plot(ax=ax,l='y2 + L1')
        # plt.show()
        return


    return (G1,G2)

def test_alg_3():



    files = [
            './instances/subproblems/sp-2-100-m_1.json',
             './instances/subproblems/sp-2-100-u_1.json'
             ]

    for level in [0,1,2]:
        L_Y_U = list()
        for i, file in enumerate(files):
            Y = PointList.from_json(file)
            # assert Y == methods.N(Y)
            Yse = PointList([y for y in Y if y.cls =='se'])
            L = methods.N(induced_LB_3d(Yse, level, PLOT=False))
            U = Yse
            L_Y_U.append((L,Y,U))

            # Y.plot(ax=ax, l=f"$Y^{i}$")
            # L.plot(ax=ax, color=Y.plot_color, marker=1)
            # U.plot(ax=ax, color=Y.plot_color, marker=2)
        # plt.show()
        # U.plot(SHOW=True)
        algorithm3_pair(L_Y_U)
    plt.show()
def main():
    
    if False:
        Y = PointList.from_json('./instances/subproblems/sp-2-10-l_1.json')
        Y = PointList([y for y in Y if y.cls =='se'][1:3])
    else:
        Y = PointList.from_json('./instances/subproblems/sp-3-10-l_1.json')
        Y = PointList([y for y in Y if y.cls =='se'][2:5])
    L = induced_LB_3d(Y, 3, PLOT=True)

    # L.plot(SHOW=True)

    # multiple_induced_UB()

if __name__ == '__main__':
    test_alg_3()   
    # main()
