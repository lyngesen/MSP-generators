import numpy as np
from classes import Point, PointList
import csv
import random
import math
import matplotlib.pyplot as plt

def generate_PointList(n,p, method = "BINOM", SAVE = False):

    def BINOM(MAXVAL = n, PROP_BINOM = 0.5):
        return np.random.binomial(MAXVAL, PROP_BINOM ,p)
    
    def DISK(radius = n):
        assert p in [1,2,3]
        if p == 1: return int(random.random()*n)
        # Generate a random angle and a random radius
        angle = 2 * math.pi * random.random()
        if p ==2:
            sqrt_radius = math.sqrt(random.random()) * radius

            # Convert polar coordinates to Cartesian coordinates
            # add integer rounding and move to positive quadrant
            x = int(sqrt_radius * math.cos(angle) + radius)
            y = int(sqrt_radius * math.sin(angle) + radius)

            return((x, y))
        else: # p == 3

            # generate a random inclination
            inclination = math.acos(2*random.random() - 1)
            cube_root_radius = random.random()**(1/3) * radius
            # convert spherical coordinates to cartesian coordinates
            x = int(cube_root_radius * math.sin(inclination) * math.cos(angle) + radius)
            y = int(cube_root_radius * math.sin(inclination) * math.sin(angle) + radius)
            z = int(cube_root_radius * math.cos(inclination) + radius)
            return((x, y, z))

            # return((x, y, z))
        



    def CONCAVE(radius = n,l = 1.5):
        """ Generates a point in the square of sidelength n*l, where points are not in the circle centered at 0 with radius n """
        x = np.random.random_sample(p) * l
        
        while np.linalg.norm(x) < 1:
            x = np.random.random_sample(p) * l

        x = tuple((int(xi*n) for xi in x))
        return((x))


    # define method
    assert method in ["BINOM", "DISK", 'CONCAVE']
    if method == "BINOM":
        random_point_generator = BINOM
        Y = PointList([Point(random_point_generator()) for _ in range(n)])
    elif method == "DISK":
        random_point_generator = DISK
        Y = PointList([Point(random_point_generator()) for _ in range(n)])
    elif method == "CONCAVE":
        random_point_generator = CONCAVE
        Y = PointList([Point(random_point_generator()) for _ in range(n)])
    return Y




def main():
    random.seed(1)
    S = 2
    P = [2]
    # P = [3]
    N = [10,20,50,100]
    methods = ["DISK", "BINOM", "CONCAVE"]
    # methods = ["CONCAVE"]
    # methods = ["DISK"]
    for p in P:
        for n in N:
            for s in range(1,S+1): 
                for method in methods:
                    Y = generate_PointList(n,p, method = method)
                    fname = f"instances/testsets/{method}-p{p}-n{n}-s{s}"
                    print(f"Generating file {fname}")
                    Y.save_csv(fname)
                    #Y.plot(fname = fname)


if __name__ == "__main__":
    # test()
    main()
