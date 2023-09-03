import numpy as np
import copy


class LPP:
    '''
        A:              Coefficient Matrix(m x n)
        b:              Constant matrix(m x 1)
        C:              Cost matrix(n x 1)
        Assumed to be a maximization problem
    '''
    def __init__(self, A,b,C):
        self.A = copy.deepcopy(A)
        self.b = copy.deepcopy(b)
        self.C = copy.deepcopy(C)

    def check(self):
        # Dimensions check
        if(self.A.shape[0]!=self.b.shape[0]):
            print("Error: A.shape[0]!=b.shape[0]")
            return False
        
        if(self.A.shape[1]!=self.C.shape[0]):
            print("Error: A.shape[1]!=C.shape[0]")
            return False
        
        if(1!=self.C.shape[1]):
            print("Error: 1!=C.shape[1]")
            return False
        
        if(1!=self.b.shape[1]):
            print("Error: 1!=b.shape[1]")
            return False
        return True



def RSM(p,BFS):
    global DEGENERACY
    '''
        Inputs:
        p  :            Linear Programming Problem
        BFS:            Initial Basic Feasible Solution(n x 1)

        Outputs:
        z,sol

        where,
        z:      the optimal value of the objective function(scalar)
        sol:    the optimal solution(n x 1)
    '''

    A = copy.deepcopy(p.A)
    b = copy.deepcopy(p.b)
    C = copy.deepcopy(p.C)

    n = A.shape[1]  # number of variables
    m = A.shape[0]  # number of equations


    sol = []
    for i in range(n):
        sol.append(-1)
    

    Xnb = set()
    Cb = []
    temp = []
    index = 0
    var_at_index = []
    for i in range(m + 1):
        var_at_index.append(0)

    for i in range(n):
        if 0==BFS[i]:
            Xnb.add(i)
        else:
            Cb.append(C[i,0])
            temp.append(A[:,i])
            var_at_index[index] = i
            index+=1




    Cb = np.matrix(Cb).transpose()

    B = []



    for i in range(1,len(temp)):
        temp[0] = np.concatenate((temp[0],temp[i]),1)

    B = temp[0]
    B_inv  = np.linalg.inv(B)
    w = np.matmul(Cb.transpose(), B_inv)
    z = np.matmul(w,b)
    b_ = np.matmul(B_inv,b)

    revised_simplex_tableau = np.matrix(np.zeros((1 + m,1 + m)))

    revised_simplex_tableau[0,0:m] = w
    revised_simplex_tableau[0,m:m+1] =  z
    revised_simplex_tableau[1:m+1,0:m] =  B_inv
    revised_simplex_tableau[1:m+1,m:m+1] =  b_

    # print(revised_simplex_tableau)

    while(True):

        # print("--------------------------------")


        w = revised_simplex_tableau[0,0:m]
        z = revised_simplex_tableau[0,m:m+1]
        B_inv = revised_simplex_tableau[1:m+1,0:m]
        b_ = revised_simplex_tableau[1:m+1,m:m+1]


        min_reduced_cost = 10**10
        nb_id = -1
        for i in Xnb:
            reduced_cost_of_i = np.matmul(w,A[:,i]) - C[i,0]
            if reduced_cost_of_i < min_reduced_cost:
                min_reduced_cost = reduced_cost_of_i
                nb_id = i
        
        if(abs(min_reduced_cost) < 10**-9):
            break
        if(min_reduced_cost > 0):
            break
        temp = np.matrix(np.zeros((1+m,1)))
        temp[0,0] = min_reduced_cost
        temp[1:m+1,0] = np.matmul(B_inv,A[:,nb_id])


        revised_simplex_tableau =  np.concatenate((revised_simplex_tableau,temp),1)

        #performing Minimum Ratio Test (MRT)

        index = -1
        minimum_ratio = 10**10
        for i in range(1, 1+m):
            if(revised_simplex_tableau[i,m + 1]<=0):
                continue

            cur_ratio = revised_simplex_tableau[i,m] / revised_simplex_tableau[i,m + 1]
            if(cur_ratio < minimum_ratio):
                minimum_ratio = cur_ratio
                index = i

        if(-1==index):
            z = "INF, (UNBOUNDED PROBLEM)"
            return z, sol
        



        Xnb.remove(nb_id)
        Xnb.add(var_at_index[index-1])
        var_at_index[index-1] = nb_id

        revised_simplex_tableau[index,:]*=(1/revised_simplex_tableau[index,m+1])
        for i in range(m+1):
            if(i==index):
                continue
            revised_simplex_tableau[i,:] = revised_simplex_tableau[i,:] - revised_simplex_tableau[index,:]*revised_simplex_tableau[i,m+1]

        revised_simplex_tableau = revised_simplex_tableau[0:m+1,0:m+1]
        
    
    for i in range(m):
        sol[var_at_index[i]] = b_[i,0]
        if b_[i,0] < 10**-9:
            DEGENERACY = True

    for i in range(n):
        if not i in var_at_index:
            sol[i] = 0 

    return z[0,0], sol



def solve(p):
    if not p.check():
        return "Incorrect LPP"
    
    M = 10**10
    
    A = copy.deepcopy(p.A)
    b = copy.deepcopy(p.b)
    C = copy.deepcopy(p.C)
    

    n = A.shape[1]  # number of variables
    m = A.shape[0]  # number of equations


    temp = np.matrix(np.zeros((m,m)))
    for i in range(m):
        temp[i,i] = 1
    A = np.concatenate((A,temp),1)
    C = np.concatenate((C,-M*np.matrix(np.ones((m,1)))),0)

    art_BFS = np.concatenate((np.matrix(np.zeros((n,1))),b),0)
    art_p = LPP(A,b,C) 


    val = RSM(art_p,art_BFS)
    if(val[0]==str):
        return val 
    return val[0], val[1][0:n]


def branch(p,x_cur, index,total):
    if(index==total):
        ok = True
        for i in range(total):
            if(x_cur[i,0]<0):
                ok=False
        if(np.all(np.matmul(p.A, x_cur)==p.b) and ok):
            return np.matmul(p.C.transpose(), x_cur)[0,0], x_cur
        else:
            return -10**10, x_cur
        
    z_max = -10**10
    sol_max = None

    x_new = copy.deepcopy(x_cur)
    x_new[index,0]+=0
    z1_max, sol1_max = branch(p,x_new,index+1,total)

    if(z1_max >= z_max):
        z_max = z1_max
        sol_max = sol1_max

    x_new = copy.deepcopy(x_cur)
    x_new[index,0]+=1
    z2_max, sol2_max = branch(p,x_new,index+1,total)

    if(z2_max >= z_max):
        z_max = z2_max
        sol_max = sol2_max

    x_new = copy.deepcopy(x_cur)
    x_new[index,0]+=2
    z3_max, sol3_max = branch(p,x_new,index+1,total)

    if(z3_max >= z_max):
        z_max = z3_max
        sol_max = sol3_max

    return z_max, sol_max
    

def solve_ILP(p,sol):
    n = p.A.shape[1]
    sol_temp = copy.deepcopy(sol)
    for i in range(n):
        sol_temp[i] = sol_temp[i]//1 - 1
    x_temp = np.matrix(sol_temp).transpose()
    val = branch(p,x_temp,0,n)
    return val



    

# adj_matrix = np.matrix([[0,1,0,0],
#                         [1,0,1,1],
#                         [0,1,0,1],
#                         [0,1,1,0]])


adj_matrix = np.matrix([[0,1,1],
                        [1,0,1],
                        [1,1,0]])


no_of_nodes = adj_matrix.shape[0]

edges = []

edges_id = {}
index = 0

for i in range(no_of_nodes):
    for j in range(i+1,no_of_nodes):
        if(1==adj_matrix[i,j]):
            edges.append([i,j])
            edges_id[(i,j)] = index
            index+=1




A = np.matrix(np.zeros((len(edges_id), no_of_nodes)))

for i in (edges_id):
    A[edges_id[i],i[0]] = 1
    A[edges_id[i],i[1]] = 1

b1 = np.matrix(np.ones((len(edges_id),1)))
A = A

temp = np.matrix(np.zeros((len(edges_id),len(edges_id))))

for i in range(no_of_nodes):
    temp[i,i] = -1
A1 = np.concatenate((A,temp),1)

C1 = np.concatenate((np.matrix(np.ones((no_of_nodes,1))) ,np.matrix(np.zeros((len(edges_id),1))) ))


C1*=-1
p1 = LPP(A1,b1,C1) # Takes input as maximization LPP
z,sol = solve(p1)



z,sol = solve_ILP(p1,sol)


z*=-1



print("Min number of nodes = ",z)
print("nodes used are ->")
for i in range(no_of_nodes):
    if(1==sol[i]):
        print(i, end=", ")
