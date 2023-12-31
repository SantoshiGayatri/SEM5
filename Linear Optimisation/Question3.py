''' 
    Team Members - 
    Tushita Sharva Janga - CS21BTECH11022
    Varshini JOnnala - CS21BTECH11024
    AKhila Kumbha - CS21BTECH11031
    Santoshi Gayatri Mavuru - CS21BTECH11036 
'''

import numpy as np
import random
import csv

visited_Z = []
visisted_Z_cost = []

# for finding +ve cost directions
def pos_cost(W, C):
    costs = np.dot(W, C)
    pos_cost_dir = np.where(costs > 0)[0]
    return pos_cost_dir


def find_feasible_point(A,B,C):
    ite = 100 
    if np.all((B >= 0)):      # if for all thr rhs value is greater than or equal to zero, it means the area in which feasible points exist, defenitely has origin
        return np.zeros(C.size)
    else:
        for j in range(ite):
            rand_indx = random.sample(range(B.size), C.size)  # selecting random n constraints out of m
            temp_A = [A[i] for i in rand_indx]
            temp_B = [B[i] for i in rand_indx]
            
            if np.linalg.det(temp_A) != 0:    # if the det is 0, we cannot find the inverse
                temp_fea_point = np.linalg.solve(temp_A, temp_B)    # finding a possible feasible point which satisfies atleast n constraints
                
                if np.all((np.dot(A, temp_fea_point) - B <= 1e-9)):     # checking whether that feasible point almost satisfies all the constraints
                    return temp_fea_point
        return None  # after maximum iterations(ite), no feasible point found
   
def remove_degeneracy(A,B,C):
    num_rows = len(B) - len(A[0])
    num = 0
    ite = 100

    while True:
        if num < ite:   # changing the values of B
            # by a little if the number of iteration is less than ite
            for i in range(num_rows):
                B[i] += 1e-9 + (1e-8 - 1e-9) * np.random.rand()
            num = num + 1
        else:
            # by a little big value if iterations exceed
            for i in range(num_rows):
                B[i] += 0.1 + (1 - 0.1) * np.random.rand()
          
        Z = find_feasible_point(A, B, C)    # calculating feasible point

        ind1 = []  # to store indices of constraints that are almost satisfied
        dot_product_result = np.dot(A, Z)
        for i in range(len(dot_product_result)):
            if abs(dot_product_result[i] - B[i]) < 1e-9:
                ind1.append(i)

        # it is degenerate (no unique solution) if no.of rows != no.of variables
        if len(ind1) == len(A[1]):
            print('\nConverted to Non-degenerate')
            break

    return B


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
    
    with open("inp1.csv", 'r') as csvfile:
        r = csv.reader(csvfile)
        inp = list(r)

    if (len(inp) < 3): 
        raise ValueError("CSV file should have at least 3 rows.")
    
    # separating the input data as per requirement
    Z = np.array([float(val) for val in inp[0][:-1]])   # initial feasible point
    C = np.array([float(val) for val in inp[1][:-1]])   # coeffs of cost function
    B = np.array([float(val) for val in [row[-1] for row in inp[2:]]])  # right hand values of coeffs
    A = np.array([[float(element) for element in row[:-1]] for row in inp[2:]])    # coeffs of constraints
    
    B = remove_degeneracy(A,B,C)  # changing into non degenerate
    
    if ((A.shape[0] < len(B)) or (A.shape[1] < len(Z))):
        raise ValueError("Matrix A should have enough rows and columns.")
    
    while True:
        V = finding_neighbour(A, B, C, Z)
        if V is None: break # If the neighbour is NONE -> current is optimal solution; else move to next one
        else:  Z = V

    answer = np.dot(C, Z)

    print('\nThe Sequence of vertices visited: ')
    for i in range(len(visited_Z)):
        print(visited_Z[i], " => Objective function Cost: ", visisted_Z_cost[i]) 
    print(f"\nThe Maximum Value of Objective Function is: {answer}\n")

# calling func
Simplex()