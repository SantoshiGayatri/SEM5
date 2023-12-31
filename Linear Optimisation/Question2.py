''' 
    Team Members - 
    Tushita Sharva Janga - CS21BTECH11022
    Varshini JOnnala - CS21BTECH11024
    AKhila Kumbha - CS21BTECH11031
    Santoshi Gayatri Mavuru - CS21BTECH11036 
'''

import numpy as np
import csv

visited_Z = []
visisted_Z_cost = []

# for finding +ve cost directions
def pos_cost(W, C):
    costs = np.dot(W, C)
    pos_cost_dir = np.where(costs > 0)[0]
    return pos_cost_dir


def finding_neighbour(A, B, C, Z):

    visited_Z.append(Z)

    visisted_Z_cost.append(np.dot(C, Z))

    ind1 = []  # to store indices of constraints that are almost satisfied
    dot_product_result = np.dot(A, Z)
    for i in range(len(dot_product_result)):
        if abs(dot_product_result[i] - B[i]) < 1e-9:
            ind1.append(i)

    ind2 = [i for i in range(len(A)) if i not in ind1]  #unsatisfied constraints

    A_new = A[ind1]   # rows of A where the linear constraints are approximately satisfied.
        
    neigh = -np.linalg.inv(A_new.T) # neighbour with greater cost    

    directions = pos_cost(neigh, C) # directions with +ve costs

    # If no +ve cost dirs => current is optimal, else +ve cost dir vector
    if (len(directions) != 0):
        
        v = neigh[directions[0]] # +ve cost dir vector

        bound_check = len(np.where(np.dot(A, v) > 0)[0])  

        if bound_check == 0:   # if no value is greater than 0, it means that we cant have a finite max value of objective func
            print('\nIt is Unbounded\n')
            exit()

        nu = B[ind2] - np.dot(A[ind2], Z)
        dem = np.dot(A[ind2], v)
        
        pos_val = dem > 0
        val = nu[pos_val] / dem[pos_val]

        maZ_feasible_step = np.min(val[val >= 0])
        final = Z + maZ_feasible_step * v    

        return final # max feasible neighbour of Z
    
    else: 
        return None

def Simplex():
    
    with open("inp.csv", 'r') as csvfile:
        r = csv.reader(csvfile)
        inp = list(r)

    if (len(inp) < 3): 
        raise ValueError("CSV file should have at least 3 rows.")
    
    # separating the input data as per requirement
    Z = np.array([float(val) for val in inp[0][:-1]])   # initial feasible point
    C = np.array([float(val) for val in inp[1][:-1]])   # coeffs of cost function
    B = np.array([float(val) for val in [row[-1] for row in inp[2:]]])  # right hand values of coeffs
    A = np.array([[float(element) for element in row[:-1]] for row in inp[2:]])    # coeffs of constraints
    
    
    if ((A.shape[0] < len(B)) or (A.shape[1] < len(Z))):
        raise ValueError("Matrix A should have enough rows and columns.")
    
    while True:
        V = finding_neighbour(A, B, C, Z)
        if V is None: break # If the neighbour is NONE -> current is optimal solution; else move to next one
        else:  Z = V

    answer = np.dot(C, Z)

    print('\nThe Sequence of vertices visited: ')
    for i in range(len(visited_Z)):
        print(visited_Z[i], " => Objective function Cost: ", visisted_Z_cost[i] ) 
    print(f"\nThe Maximum Value of Objective Function is: {answer}\n")

# calling func
Simplex()