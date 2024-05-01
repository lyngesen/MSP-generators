from classes import Point, PointList, LinkedList, MinkowskiSumProblem
import methods
from methods import N
import math
import matplotlib.pyplot as plt
import matplotlib
import pyomo.environ as pyomo
from timing import timeit, print_timeit

# def buildModel(Y_list: list[PointList],) -> pyomo.ConcreteModel():
    # pass


@timeit
def build_model(Y_list) -> pyomo.ConcreteModel():
    print(f"Setting up model..")
    P = Y_list[0][0].dim
    assert all((P == Y.dim for Y in Y_list))

    Yn = methods.MS_sequential_filter(Y_list)

    S = list(range(len(Y_list)))
    J = list(range(len(Yn)))
    I_dict = {s : list(range(len(Y_list[s]))) for s in S}

    # Define your model
    model = pyomo.ConcreteModel()

    # Define your sets
    model.S = pyomo.Set(initialize=S)
    model.J = pyomo.Set(initialize=J)
    model.I_s = pyomo.Set(model.S, initialize=I_dict)

    # Create an intermediate set
    def IS_rule(model):
        return [(s, i) for s in S for i in model.I_s[s]]

    model.IS = pyomo.Set(within=model.S*model.J, initialize=IS_rule)

    # Define binary variables
    model.x = pyomo.Var(model.IS, within=pyomo.Binary)
    model.w = pyomo.Var(model.IS, model.J, within=pyomo.Binary)

    # Define constraints
    model.cons = pyomo.ConstraintList()
    for s in S: # at least one solution to each subproblem
        model.cons.add(sum(model.x[s, i] for i in model.I_s[s]) >= 1)
        # at most one part of j can be composed of points in subproblem s
        for j in model.J:
            model.cons.add(sum(model.w[s, i, j] for i in model.I_s[s]) == 1)
        

    for s, i in model.IS: 
        for j in model.J:
            model.cons.add(model.w[s, i, j] <= model.x[s, i])
    
    for j in model.J:
        for p in range(P):
            model.cons.add(sum(model.w[s, i, j]*Y_list[s][i][p] for s in S for i in I_dict[s]) - Yn[j][p]  == 0)
    
    # Define objective
    def objective_rule(model):
        return sum(model.x[s, i] for s, i in model.IS)

    model.obj = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)

    return model


@timeit
def solve_model(model:pyomo.ConcreteModel(), solver_str = "cplex_direct"):
    assert solver_str in ["cbc", "cplex_direct", "plpk"]
    print(f"Solving model..")
    # Solve model
    solver = pyomo.SolverFactory(solver_str)
    solver.solve(model, tee = True)


@timeit
def retrieve_solution(model:pyomo.ConcreteModel(), Y_list: list[PointList], verbose = False) -> list[PointList]:
    # Retrieve solution
    xVal =[]
    Y_generator_list = []
    for s in model.S:
        Y_generator = []
        if verbose:
            print(f"{s=}")
        for i in model.I_s[s]:
            if verbose:
                print('    x[{}]={}'.format(i, pyomo.value(model.x[s, i])))
            if pyomo.value(model.x[s,i]) == 1:
                Y_generator.append(Y_list[s][i])
            elif math.isclose(pyomo.value(model.x[s,i]),1):
                if verbose:
                    print(f"  math.isclose() rounding used..")
                Y_generator.append(Y_list[s][i])
        Y_generator_list.append(PointList(Y_generator))

    return Y_generator_list


    
@timeit
def display_solution(Y_list: list[PointList], Y_generator_list: list[PointList], verbose = True, plot = True):
    # Display solution
    if verbose:
        total_standard, total_generator = 0, 0
        if verbose == "all":
            print("_"*55)
        for s, _ in enumerate(Y_list):
            total_standard += len(Y_list[s])
            if verbose == "all":
                print(f"|Y{s}| = {len(Y_list[s])}")
                print(f"|^Y{s}| = {len(Y_generator_list[s])}")
            total_generator += len(Y_generator_list[s])
        print("_"*55)
        print(f"SUM|Y| = {total_standard}")
        print(f"SUM|^Y| = {total_generator}")
        print("_"*55)

    if plot:
        plt.style.use("ggplot")
        Yn_all = methods.MS_sum(Y_list)
        Yn_all.plot("Y1n + Y2n +...", color="gray")


        for s, Y in enumerate(Y_list):
            Y.plot(l=f"Y{s}", marker = "o")
        for s, Y_generator in enumerate(Y_generator_list):
            Y_generator.plot(l=f"^Y{s}", marker = f"{s+1}")
        

        Yn = methods.MS_sequential_filter(Y_list)
        Yn.plot(l="Yn")

        Yn_hat = methods.MS_sum(Y_generator_list)
        assert N(Yn_hat.removed_duplicates()) == N(Yn.removed_duplicates())
        Yn_hat.plot(l="^Yn", marker= "3")

        plt.show()



def solve_instance(Y_list: list[PointList], verbose = 'all', plot = False):
    model = build_model(Y_list)
    solve_model(model)
    Y_MIN_LIST = retrieve_solution(model, Y_list)
    if verbose or plot:
        display_solution(Y_list, Y_MIN_LIST, verbose = verbose, plot = plot)

    return Y_MIN_LIST




def SimpleFilter(Y1, Y2):
    pass

def main():

    # setup test instance
    Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s1")
    # Y2 = PointList.from_csv("instances/testsets/CONCAVE-p2-n10-s2")
    Y2 = PointList.from_csv("instances/testsets/BINOM-p2-n20-s1")
    Y3 = PointList.from_csv("instances/testsets/CONCAVE-p2-n20-s2")
    Y4 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s2")
    Y2 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s2")
    # Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n20-s1")
    # Y3 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s1")
    # Y4 = PointList.from_csv("instances/testsets/DISK-p2-n10-s1")
    
    Y_list = [Y1, Y2, Y3, Y4] 
    # Y_list = [N(Y.removed_duplicates()) for Y in Y_list]
    Y_list = [N(Y) for Y in Y_list]
 
    # problem_instance = "instances/problems/prob-2-50|50|50-ull-3_5.json" 
    # print(f"Reading MSP problem {problem_instance}")
    # MSP = MinkowskiSumProblem.from_json(problem_instance)
    # print(f"{MSP=}")
    # Y_list =  MSP.Y_list

    solve_instance(Y_list, verbose = 'all', plot = True )

    print_timeit()

if __name__ == "__main__":
    main()  
