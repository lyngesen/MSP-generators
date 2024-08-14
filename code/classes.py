from __future__ import annotations # allow annotation self references (eg. in KD_Node)
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import json
import os

import pprint
import traceback
import sys
# from itertools.collections import Counter
import collections


"""
Class
@Point

y1 = Point((4,2))
y2 = Point([3,2])

y1 < y2
>> False
t2 <= y1 
>> True


Class
@PointList 

Y1 = PointList((y1, y2))
Y2 = PointList((y2))
Y3 = PointList.from_csv(fname)

Y1 == Y2
>> False, since counter is off Y1:{y1: 1, y2: 1}, while Y2: {y2: 1}

Y3.save_csv(fname)
>> saves the list to a csv file with name fname

Y1.plot()
> plots set of points, if True plt.show() is called

Y2.dominates_point(y1)
>> True if the point y1 is dominated by the set Y2


"""

@dataclass
class Point:
    val: np.array(iter)
    dim = None
    plot_color = None
    cls = None

    def __post_init__(self):
        if not isinstance(self.val, np.ndarray):
            self.val = np.array(self.val)
        if self.dim == None:
            self.dim = len(self.val)
    def __lt__(self, other):
        if all(self.val == other.val):
            return False
        return all(self.val <= other.val)

    def __le__(self, other):
        return all(self.val <= other.val)

    def le_d(self, other, d : int):
        return all((self.val[p] <= other.val[p] for p in range(d)))
    
    def lt_d(self, other, d : int):
        if all((self.val[p] == other.val[p] for p in range(d))):
            return False
        return all((self.val[p] <= other.val[p] for p in range(d)))

    def strictly_dominates(self, other):
        return all(self.val < other.val)

    def lex_le(self, other):
        if len(self.val) == 2:
            if self.val[0] > other.val[0]:
                return False
            if self.val[0] < other.val[0]:
                return True
            if self.val[0] == other.val[0] and self.val[1] > other.val[1]:
                return False
            else:
                return True
        if len(self.val) > 2:
            for p in range(self.dim):
                if self[p] < other[p]:
                    return True
                elif self[p] > other[p]:
                    return False
            return True

    def __gt__(self, other):
        if all(self.val == other.val):
            return False
        return all(self.val >= other.val)
    def __iter__(self):
        return self.val.__iter__()
    def __hash__(self):
        return tuple(self.val).__hash__()
    def __eq__(self, other):
        return (self.val == other.val).all()
    def __repr__(self):
        return tuple(self.val).__repr__()
    def __getitem__(self, item):
        return self.val[item]
    def __add__(self, other):
        if isinstance(other, PointList):
            return PointList((self,)) + other
        return Point(self.val + other.val)

  
    def __sub__(self, other):
        return Point(self.val - other.val)
    def __mul__(self, other):
        if isinstance(other, int):
            return Point(self.val * other)
        elif isinstance(other, float):
            return Point(self.val * other)
        elif isinstance(other, Point):
            return Point(self.val * other.val)
        else:
            raise TypeError(f'__mul__ not implemented for {type(other)=}')
    

    def plot(self, SHOW = False, fname = None, ax = None, l =None,label_only = False, color = None,  **kwargs):
        assert self.dim<=3, 'Not implemented for p > 3'
        ax = ax if ax else plt
        color = color if (color is not None) else self.plot_color
        kwargs['color'] = color
        if self.dim == 3: 
            ax.scatter = ax.scatter3D

        if not label_only:
            plot = ax.scatter(*self.val, **kwargs)
            self.plot_color = plot.get_facecolor()
        if l != None:
            if self.dim == 3:
                ax.text(*self.val, l)
            else:
                ax.annotate(text=l, xy= self.val, xytext=self.val*1.02 )
                
        if l != None:
            ax.legend(loc="upper right") 
        if fname:
            ax.savefig(fname, dpi= 200)
            ax.cla()
        if SHOW:
            ax.show()
        return ax 

    def plot_cone(self, ax= None, quadrant = 1,color='darkgray', **kwargs):
        assert self.dim<=2, 'plot_cone Not implemented for p > 2'
        ax = ax if ax else plt
        color = color if (color is not None) else self.plot_color
        kwargs['color'] = color
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        if quadrant == 1:
            ax.add_patch(Rectangle((self[0],self[1]), xmax-self[0], ymax-self[1], fill=False, hatch='xx', **kwargs))
        if quadrant == 3:
            ax.add_patch(Rectangle((xmin,ymin), self[0]- xmin, self[1] - ymin, fill=False, hatch='xx', **kwargs))
        return ax

@dataclass
class PointList:
    points: tuple[Point] = ()
    dim = None
    plot_color = None
    statistics : dict = None
    filename = None
    np_array : np_array = None
    def __post_init__(self):
        # Check if SINGLETON: allows for PointList((y)) where y is of class Point 
        if isinstance(self.points, Point):
            self.points = (self.points,)
        else: #unpack list
            self.points = tuple([y if isinstance(y, Point) else Point(y) for y in self.points])
        if self.points:
            self.dim = self.points[0].dim

        self.statistics = {
            "p": [self.dim],
            "card": [len(self.points)],
            "supported": [None],
            "extreme": [None],
            "unsupported": [None],
            "min": [None],
            "max": [None, None],
            "width": [None, None],
            "method": [None],
          }

    def __iter__(self) -> tuple[Point]:
        return tuple(self.points).__iter__()
    def __len__(self):
        return tuple(self.points).__len__()
    
    def plot(self,  l =None,SHOW = False, fname = None, ax= None, line=False, color = None, point_labels = False, **kwargs):
        ax = ax if ax else plt
        assert self.dim<=3, 'Not implemented for p > 3'
        # color = self.plot_color if (color is not None) else color
        color = color if (color is not None) else self.plot_color
        kwargs['color'] = color
        
        if self.dim == 3: 
            ax.scatter = ax.scatter3D
            ax.plot = ax.plot3D

        if line:
            plot = ax.plot(*zip(*self.points), label =l, **kwargs)
            self.plot_color = plot[-1].get_color()
        else:
            plot = ax.scatter(*zip(*self.points), label =l, **kwargs)
            # self.plot_color = plot.to_rgba(-1) # save used color to object
            self.plot_color = plot.get_facecolors()
        if l:
            ax.legend(loc="upper right") 
        if fname:
            plt.savefig(fname, dpi= 200)
            plt.cla()
        if point_labels:
            if point_labels == True:
                point_labels = ["$y^{" +  f"{i}" + "}$" for i, _ in enumerate(self, start = 1)]
            # add labels to points
            for i,y in enumerate(self):
                # y.plot(ax = ax, l= "$y^{" +  f"{i}" + "}$", label_only=True)
                y.plot(ax = ax, l= point_labels[i], label_only=True)
           

                
        if SHOW:
            ax.show()

        return ax

    def dominates_point(self, point:Point):
        for y in self.points:
            if y < point:
                # return True
                return y
        return False

    def weakly_dominates_point(self, point:Point):
        for y in self.points:
            if y <= point:
                return y
                # return True
        return False



    def __add__(self,other):
        """
        input: list of two PointList
        output: Minkowski sum of sets
        """
        return PointList([y1 + y2 for y1 in self for y2 in other])
    
    def __sub__(self,other):
        """
        input: list of two PointList
        output: Minkowski subtration of sets
        """
        return PointList([y1 - y2 for y1 in self for y2 in other])


    def __mul__(self,other):
        """
        input: list of two PointList
        output: Minkowski subtration of sets
        """
        match other:
            case ( float() | int() ):
                return PointList([y*other for y in self])
            case _:
                print(f"{other=}")
                print(f"{type(other)=}")
                raise NotImplementedError




    def get_nadir(self):
        nadir_vals = list(tuple(self.points[0].val))
        for point in self.points:
            for p in range(self.dim):
                if nadir_vals[p] < point.val[p]:
                    nadir_vals[p] = point.val[p]
        self.nadir = Point(nadir_vals)
        return self.nadir
    def get_ideal(self):
        ideal_vals = list(tuple(self.points[0].val))
        for point in self.points:
            for p in range(self.dim):
                if ideal_vals[p] > point.val[p]:
                    ideal_vals[p] = point.val[p]
        self.ideal = Point(ideal_vals)
        return self.ideal



    def dominates(self, other, power="default"):
        match power:
            case "default":
                if self == other:
                    return False
                for y in other.points:
                    if any((l <= y for l in self.points)):
                        continue
                    else:
                        return False
                return True

            case "strict":
                for y in other.points:
                    if any((l < y for l in self.points)):
                        continue
                    else:
                        return False
                return True


    def save_csv(self, filename="testsets/disk.csv"):
        with open(f"{filename}", "w") as out:
            csv_out=csv.writer(out)
            for y in self.__iter__():
                csv_out.writerow(y)   


    def save_raw(self, filename : str):
        # raw format used in c-interface
        with open(filename, 'w') as out:
            out.write(f"{self.statistics['p'][0]}" + "\n")
            out.write(f"{self.statistics['card'][0]}" + "\n")
            for y in self.__iter__():
                out.write(" ".join([f"{yp:.6f}" for yp in y]) + "\n")

    def from_raw(filename : str):
        # raw format used in c-interface
        with open(filename, "r") as rawfile:
            dim = int(rawfile.readline())
            n = int(rawfile.readline())
            lines = rawfile.read().splitlines()
        y_list = []
        for line in lines:
            y = Point([float(yp) for yp in line.split(' ') if yp != ''])
            y_list.append(y)
        return PointList(y_list)

    def from_csv(filename = "disk.csv"):
        with open(f"{filename}", "r") as csvfile:
            points = []
            for y in csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC):
                points.append(Point(y))
            return PointList(points)

    def as_dict(self):

        PointList_dict = {
            "points":
                          [dict({f"z{p+1}": point[p] for p in range(point.dim)},**({'cls':None})) for point in self.points],
            'statistics': self.statistics
          }
        return PointList_dict 
    
    def as_np_array(self):
        # if self.np_array is not None:
        # if isinstance(self.np_array, NoneType):
            # print(f"bla bla")
            # self.np_array = np.array([y.val for y in self.points])
        return np.array([y.val for y in self.points])
        # return self.np_array

    def save_json(self, filename, max_file_size = 100):
        json_str = json.dumps(self.as_dict(), indent=None, separators=(',', ':'))
        # Calculate size (approx) in bytes
        size_mb = len(json_str.encode('utf-8')) / 1_000_000
        if size_mb >= max_file_size:
            if True:
                print(f"*** {filename}, pointlist with {len(self)} points too large for json. Saving raw format. ESTIMATED MB {size_mb} > {max_file_size}=MAX")
                self.save_raw(filename.replace('.json','.raw'))
                # return
            if True:
                # print(f"*** {filename}, pointlist with {len(self)} points too large for json. Saving json without points. ESTIMATED MB {size_mb} > {max_file_size}=MAX")
                json_str = self.as_dict()
                json_str['points'] = []
                json_str = json.dumps(json_str, indent=None, separators=(',', ':'))
        with open(filename, 'w') as json_file:
            json_file.write(json_str)
            # json.dump(self.as_dict(), json_file)




    def from_json_str(json_dict):
        statistics = json_dict['statistics']
        points = []
        for json_point in json_dict['points']:
            values = [json_point[f"z{p+1}"] for p in range(statistics["p"][0])]
            # TODO: Error if values not casted to float65  - does not work for int64? <08-02-24> #
            values = np.float64(values)
            point = Point(values)
            if 'cls' in json_point:
                point.cls = json_point['cls']
            else:
                point.cls = None
            points.append(point)
        Y = PointList(points)
        Y.statistics = statistics

        return Y

    def from_json(filename: str):

        with open(filename, 'r') as json_file:
            json_dict = json.load(json_file)

        return PointList.from_json_str(json_dict)
        
        


    def print_data(self):
        N_POINTS = len(self.points)
        print(f"{N_POINTS=}")

    def __eq__(self, other):
        return collections.Counter(self.points) == collections.Counter(other.points)

    def __lt__(self, other):
        """
        input: two PointLists
        output: return True if each point of other is dominated by at least one point in self
        """
        for y2 in other:
            for y1 in self:
                if y1 < y2:
                    break
            else: # finally, if for loop finishes normaly
                return False
        return True
    
    

    def __getitem__(self, item):
        return self.points[item]

    def removed_duplicates(self):
        return PointList(set(self.points))



@dataclass
class MinkowskiSumProblem:
    Y_list: tuple[PointList]
    filename : str = None
    dim : int = None
    S : int = None
    
    def __post_init__(self):
        self.S = len(self.Y_list)

    def from_json(filename: str):
        with open(filename, 'r') as json_file:
            json_dict = json.load(json_file)[0]

        Y_list = []
        for V, Y_filename in sorted(json_dict.items()):
            if isinstance(Y_filename, str):
                Y = PointList.from_json("instances/" + Y_filename)
            else:
                Y = PointList.from_json_str(Y_filename)
            Y_list.append(Y)

        MSP = MinkowskiSumProblem(Y_list)
        MSP.filename = filename
        MSP.dim = Y_list[0].dim
        return  MSP

    def save_json(self, filename):
        out_dict = [
                {f"V{s}":Y.as_dict() for s,Y in enumerate(self.Y_list, start=1)},
                self.statistics
                ]
        json_str = json.dumps(out_dict, indent=1, separators=(',', ':'))
        with open(filename, 'w') as json_file:
            json_file.write(json_str)



    def from_subsets(filenames : iter[str]):
        Y_list = []
        sizes =  '|'
        method = ''
        for Y_filename in filenames:
            Y = PointList.from_json("./instances/subproblems/" + Y_filename)
            Y_list.append(Y)
            sizes += Y_filename.split('-')[2] + '|'
            method += Y_filename.split('-')[3].split('.')[0]
        filename = f'MSP-special-{sizes}-{method}'
        MSP = MinkowskiSumProblem(Y_list)
        MSP.filename = filename
        MSP.dim = Y_list[0].dim
        return  MSP

    def __repr__(self):
        string = f"MSP( filename={self.filename.split('/')[-1]}, dim={self.dim}, "
        for s,Y in enumerate(self.Y_list):
            string+=f"|Y{s+1}|={len(Y)} "

        string += ")"
        return string


    def plot(self,  hidelabel = False, ax = None, **kwargs):
        ax = ax if ax else plt
        for s, Y in enumerate(self.Y_list):
            Y.plot(l= "_"*hidelabel + "$\mathcal{Y}_{\mathcal{N}}^{" + str(s+1) + "}$", ax = ax, **kwargs)


@dataclass
class MSPInstances:
    preset : str = 'all' 
    options : dict = None
    filename_list : list[str] = None
    max_instances : int = 0 
    m_options : tuple[int]= (2,3,4,5) # subproblems
    p_options : tuple[int]= (2,3,4,5) # dimension
    generation_options : tuple[str]= ('l','m','u', 'ul') # generation method
    ignore_ifonly_l : bool = False # if true ignore MSP where method i only l
    size_options : tuple[int]= (50, 100, 150, 200, 300,600) # subproblems size
    seed_options : tuple[int]=  (1,2,3,4,5)


    def instance_name_dict(problem_file):
        filename = problem_file
        problem_file = problem_file.split(".json")[0]
        problem_file, seed = problem_file.split("_")
        _, p, size, method, M = problem_file.split("-")
        size = size.split("|")[0]
        p, M, size, seed = int(p), int(M), int(size), int(seed)
        D = {'filename': filename, 'p': p, 'method':method, 'M': M, 'size': size, 'seed':seed}
        return D
        
    def instance_name_dict_keys(problem_file):
        D = MSPInstances.instance_name_dict(problem_file)
        return ( D['M'],D['size'], D['p'], D['seed'])

    def __post_init__(self):
        all_problems = os.listdir("instances/problems/")
        # print(f"{all_problems=}")
        all_problems = sorted(all_problems, key = MSPInstances.instance_name_dict_keys )
        
        self.filename_list = []

        match self.preset:
            case 'all':
                pass
            case '2d':
                self.p_options = (2,)
            case 'algorithm1':
                self.generation_options = ['m','u','l'] # generation method
                self.size_options = (50, 100, 150, 200, 300) # subproblems size
            case 'grendel_test':
                self.max_instances = 4
                self.filename_list = [
                        'prob-2-100|100-ll-2_1.json',
                        'prob-4-100|100-ll-2_1.json',
                        'prob-4-100|100|100-lll-3_1.json',
                        'prob-4-200|200|200|200|200-lllll-5_5.json'
                        ]
            case 'algorithm2':
                self.generation_options = ['m','u', 'l'] # generation method
                # self.p_options = (4,)
                # self.m_options = (4,)
                self.size_options = (50, 100, 150, 200, 300) # subproblems size
            case 'algorithm2_test':
                self.seed_options = (0,) # ignora alle other test problems
                subsets_list = []
                subsets_list.append(('sp-2-10-u_1.json', 'sp-2-10-u_1.json', 'sp-2-10-u_2.json'))
                subsets_list.append(('sp-2-50-u_1.json', 'sp-2-50-u_1.json', 'sp-2-10-u_1.json'))
                subsets_list.append(('sp-2-100-u_1.json', 'sp-2-100-l_1.json', 'sp-2-100-u_1.json'))
                subsets_list.append(('sp-4-100-u_1.json', 'sp-4-100-l_1.json', 'sp-4-100-u_1.json'))
                subsets_list.append(('sp-4-100-u_2.json', 'sp-4-100-l_1.json', 'sp-4-100-u_2.json'))
                for subsets in subsets_list:
                    self.filename_list.append(MinkowskiSumProblem.from_subsets(subsets))
            case _:
                print(f"preset '{self.preset}' not recognised")
                raise NotImplementedError
    
        for filename in all_problems:
            instance_dict = MSPInstances.instance_name_dict(filename)

            if self.ignore_ifonly_l and set(instance_dict['method']).issubset(set(('l',))):
                continue
            if all((instance_dict['p'] in self.p_options,
                   instance_dict['M'] in self.m_options,
                   set(instance_dict['method']).issubset(set(self.generation_options)),
                   instance_dict['size'] in self.size_options,
                   instance_dict['seed'] in self.seed_options,
                    (self.preset != 'algorithm1' or (not (instance_dict['p'] == 5 and instance_dict['M'] == 5 ))) # if algorithm 1 then not p=m=5
                   )):
                self.filename_list.append(filename)
            
        # limit number of files
        if self.max_instances:
            self.filename_list = self.filename_list[:self.max_instances]

    def filter_out_solved(self, save_prefix : str, solved_folder : str):
        self.not_solved = []
        self.solved = []
        for p in self.filename_list:
            if save_prefix + p in os.listdir(solved_folder):
                self.solved.append(p)
            else:
                self.not_solved.append(p)

        print(f"|solved| = {len(self.solved)}    |not solved| = {len(self.not_solved)}")

        self.filename_list = self.not_solved


    def __repr__(self):
        return f"TestInstances(size='{len(self.filename_list)}', preset='{self.preset}')"

    def __iter__(self) -> iter[MinkowskiSumProblem]:
        return (filename if isinstance(filename, MinkowskiSumProblem) else MinkowskiSumProblem.from_json('./instances/problems/' + filename) for filename in self.filename_list)

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

    def __repr__(self):
        return f"{str(self.data)}"

class LinkedList:
    def __init__(self):
        self.head = None

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(str(node.data))
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)

    def __iter__(self):
        node = self.head
        # while node is not None:
        while node is not None:
            yield node
            node = node.next

    def add_first(self, node):
        node.next = self.head
        self.head = node
        self.prev = None

    def add_after(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        for node in self:
            if node.data == target_node_data:
                new_node.next = node.next
                node.next = new_node
                return

        raise Exception("Node with data '%s' not found" % target_node_data)


    def add_before(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            return self.add_first(new_node)

        prev_node = self.head
        for node in self:
            if node.data == target_node_data:
                prev_node.next = new_node
                new_node.next = node
                return
            prev_node = node

        raise Exception("Node with data '%s' not found" % target_node_data)


    def remove_node(self, target_node_data):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            self.head = self.head.next
            return

        previous_node = self.head
        for node in self:
            if node.data == target_node_data:
                previous_node.next = node.next
                return
            previous_node = node

        raise Exception("Node with data '%s' not found" % target_node_data)


@dataclass
class KD_Node:
    y : Point
    l : int
    parent : KD_Node  = None
    LEFT : KD_Node = None
    RIGHT : KD_Node = None
    UB : Point = None
    LB : Point = None
    
    # def __repr__(self):
        # return f"{str(self.y)}"
    def __str__(self, level=0):
        ret = "\t"*level+repr(self.y) + f"l={self.l}" +"\n"
        
        if self.LEFT != None:
            ret += self.LEFT.__str__(level+1)
        else:
            ret += "\t"*(level+1) + "Ø \n"

        if self.RIGHT != None:
            ret += self.RIGHT.__str__(level+1)
        else:
            ret += "\t"*(level+1) + "Ø \n"

        return ret

    def __repr__(self):
        # return str(self.y)
        return f"KD_NODE(y={self.y}, parent={self.parent.y if self.parent else 'Ø'}, LEFT={self.LEFT.y if self.LEFT else 'Ø'}, RIGHT={self.RIGHT.y if self.RIGHT else 'Ø'}, UB = {self.UB}, LB = {self.LB})"

@dataclass
class KD_tree:
    def dominates_point_recursion(r : KD_Node, p : Point):
        # seperated for timing purposes

        if r.y <= p: return True
        if r.LEFT != None and p > r.LEFT.LB:
            return KD_tree.dominates_point_recursion(r.LEFT, p)
        if r.RIGHT != None and p > r.RIGHT.LB:
            return KD_tree.dominates_point_recursion(r.RIGHT, p)
        return False
     
    def dominates_point(r : KD_Node, p : Point):
        """ checks if point is dominated by the KD-tree rooted at r 

        Args:
            p (Point): point

        Returns: 
            1, if p is dominated by a point in the KD-tree rooted at r, 
            0, otherwise

        """
        return KD_tree.dominates_point_recursion(r,p)

    def get_UB(r : KD_Node,  p: Point):
        return Point(np.maximum(r.UB.val, p.val))
        # old
        # return Point([max(r.UB[i], p[i]) for i in range(p.dim)])

    def get_LB(r: KD_node, l : int,  p: Point):
        return Point(np.minimum(r.LB.val, p.val))
        # return Point([min(r.LB[i], p[i]) for i in range(p.dim)])

    def insert_recursion(r : KD_Node, l : int, p: Point):
        # seperated for timing purposes
        # update r.UB, r.LB
        r.UB = KD_tree.get_UB(r,p)
        r.LB = KD_tree.get_LB(r,l,p) 
        
        # compare l-th component of p and r
        # print(f"{r,l,p =}")
        if p[l] < r.y[l]:
            if r.LEFT == None:
                r.LEFT = KD_Node(p, (l + 1) % p.dim, r, UB = p, LB = p)
            elif r.LEFT != None:
                KD_tree.insert_recursion(r.LEFT, (l + 1) % p.dim, p)
        elif p[l] > r.y[l]:
            if r.RIGHT == None:
                r.RIGHT = KD_Node(p, (l + 1) % p.dim, r, UB = p, LB = p)
            elif r.RIGHT != None:
                KD_tree.insert_recursion(r.RIGHT, (l + 1) % p.dim, p)

    def insert(r : KD_Node, l : int, p: Point):
        return KD_tree.insert_recursion(r,l,p)
            
