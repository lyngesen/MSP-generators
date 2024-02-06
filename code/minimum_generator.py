from classes import Point, PointList, LinkedList
import methods
from methods import N
import math
import matplotlib.pyplot as plt
import matplotlib

import pyomo.environ as pyomo

# def buildModel(Y_list: list[PointList],) -> pyomo.ConcreteModel():
    # pass

def main():

    # setup test instance
    Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s1")
    # Y2 = PointList.from_csv("instances/testsets/CONCAVE-p2-n10-s2")
    Y2 = PointList.from_csv("instances/testsets/BINOM-p2-n20-s1")
    Y3 = PointList.from_csv("instances/testsets/CONCAVE-p2-n20-s2")
    Y4 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s2")
    # Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s2")
    # Y1 = PointList.from_csv("instances/testsets/CONCAVE-p2-n20-s1")
    # Y3 = PointList.from_csv("instances/testsets/CONCAVE-p2-n100-s1")
    # Y4 = PointList.from_csv("instances/testsets/DISK-p2-n10-s1")
    
    Y_list = [Y1, Y2, Y3, Y4] 
    Y_list = [N(Y.removed_duplicates()) for Y in Y_list]
    
    # formulate
        # input Y_list
    # solve
    # display solution
    


    Yn_all = methods.MS_sum(Y_list)

    P = Y1[0].dim
    assert all((P == Y.dim for Y in Y_list))

    Yn = methods.MS_sequential_filter(Y_list)

    S = list(range(len(Y_list)))
    J = list(range(len(Yn)))
    I_dict = {s : list(range(len(Y_list[s]))) for s in S}

    print(f"Setting up model..")
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


    # Solve model
    print(f"Solving model..")
    solver_str = "cplex_direct" # in "cbc", "cplex_direct", "plpk"
    solver = pyomo.SolverFactory(solver_str)
    solver.solve(model)

    
    # Display solution
    plt.style.use("ggplot")
    Yn_all.plot("Y1n + Y2n +...", color="gray")

    xVal =[]
    Y_hat_list = []
    for s in S:
        Y_hat = []
        print(f"{s=}")
        for i in I_dict[s]:
            print('    x[{}]={}'.format(i, pyomo.value(model.x[s, i])))
            if pyomo.value(model.x[s,i]) == 1:
                Y_hat.append(Y_list[s][i])
            elif math.isclose(pyomo.value(model.x[s,i]),1):
                print(f"  math.isclose() rounding used..")
                Y_hat.append(Y_list[s][i])
        Y_hat_list.append(PointList(Y_hat))


    for s, Y in enumerate(Y_list):
        Y.plot(l=f"Y{s}", marker = "o")
    for s, Y_hat in enumerate(Y_hat_list):
        Y_hat.plot(l=f"^Y{s}", marker = f"{s+1}")
    


    Yn.plot(l="Yn")

    Yn_hat = methods.MS_sum(Y_hat_list)
    assert N(Yn_hat.removed_duplicates()) == N(Yn.removed_duplicates())
    Yn_hat.plot(l="^Yn", marker= "3")


    total_standard, total_generator = 0, 0
    print("*"*30)
    for s in S:
        print(f"|Y{s}| = {len(Y_list[s])}")
        total_standard += len(Y_list[s])
        print(f"|^Y{s}| = {len(Y_hat_list[s])}")
        total_generator += len(Y_hat_list[s])
    print("*"*30)
    print(f"SUM|Y| = {total_standard}")
    print(f"SUM|^Y| = {total_generator}")
    print("*"*30)


    plt.show()
    # for s, i in model.IS:



if __name__ == "__main__":
    main()  
