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


    # BFS check
    # if(BFS.min() < 0):
    #     print("Error: BFS has negative values")
    #     return 
    

    
    # print(np.any(np.matmul(A,BFS)!=b))
    # if(np.any(np.matmul(A,BFS)!=b)):
    #     print("Error: np.matmul(A,BFS)!=b")
    #     return
    


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

        # print("w = ")
        # print(w)
        # print("z = ")
        # print(z)
        # print("B_inv = ")
        # print(B_inv)
        # print("b_ = ")
        # print(b_)

        # print("RST")
        # print(revised_simplex_tableau)


        min_reduced_cost = 10**10
        nb_id = -1
        for i in Xnb:
            reduced_cost_of_i = np.matmul(w,A[:,i]) - C[i,0]
            # print("non basic variable -> ",i)
            # print("reduced cost = ", reduced_cost_of_i)
            if reduced_cost_of_i < min_reduced_cost:
                min_reduced_cost = reduced_cost_of_i
                nb_id = i
        
        # print("min cost -> ",min_reduced_cost)
        # print("non basic id -> ",nb_id)
        if(min_reduced_cost > 0):
            break
        temp = np.matrix(np.zeros((1+m,1)))
        temp[0,0] = min_reduced_cost
        temp[1:m+1,0] = np.matmul(B_inv,A[:,nb_id])


        # max_reduced_cost = 10**10
        # nb_id = -1
        # for i in Xnb:
        #     reduced_cost_of_i = np.matmul(w,A[:,i]) - C[i,0]
        #     if reduced_cost_of_i < max_reduced_cost:
        #         max_reduced_cost = reduced_cost_of_i
        #         nb_id = i
        
        # if(max_reduced_cost > 0):
        #     break
        # temp = np.matrix(np.zeros((1+m,1)))
        # temp[0,0] = max_reduced_cost
        # temp[1:m+1,0] = A[:,nb_id]


        revised_simplex_tableau =  np.concatenate((revised_simplex_tableau,temp),1)

        # print("appended table -> ")
        # print(revised_simplex_tableau)


        #performing Minimum Ratio Test (MRT)

        index = -1
        minimum_ratio = 10**10
        for i in range(1, 1+m):
            if(revised_simplex_tableau[i,m + 1]<=0):
                continue

            cur_ratio = revised_simplex_tableau[i,m] / revised_simplex_tableau[i,m + 1]
            # print("basic index = ",i)
            # print("cur ration = ",cur_ratio)
            if(cur_ratio < minimum_ratio):
                minimum_ratio = cur_ratio
                index = i

        if(-1==index):
            z = "INF, (UNBOUNDED PROBLEM)"
            return z, sol
        



        # print("basic index = ",index)
        # print("basic id = ",var_at_index[index-1])
        Xnb.remove(nb_id)
        Xnb.add(var_at_index[index-1])
        var_at_index[index-1] = nb_id
        # print("non basic set = ",Xnb)

        revised_simplex_tableau[index,:]*=(1/revised_simplex_tableau[index,m+1])
        for i in range(m+1):
            if(i==index):
                continue
            revised_simplex_tableau[i,:] = revised_simplex_tableau[i,:] - revised_simplex_tableau[index,:]*revised_simplex_tableau[i,m+1]

        revised_simplex_tableau = revised_simplex_tableau[0:m+1,0:m+1]
        
        # print(Xnb)
        # print("updated RST = ")
        # print(revised_simplex_tableau)

    

    # print("final revised_simplex_tableau = \n",revised_simplex_tableau,"\n")
    # print("\nnon basic variables having value 0 ->",Xnb)
    # print("\nbasic varibles indices ->\n")
    # for i in range(m):
    #     print(str(var_at_index[i]) + " -> " + str(b_[i,0] ))
    # print("\nObjective function value = " + str(z[0,0]))

    
    for i in range(m):
        sol[var_at_index[i]] = b_[i,0]
        if b_[i,0] < 10**-9:
            DEGENERACY = True

    for i in range(n):
        if not i in var_at_index:
            sol[i] = 0 

    return z[0,0], sol


# def find_bfs(p):

#     # adding artifical variables and then solving new LPP to get BFS for old LPP
    
    

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


    # av = []
    # for i in range(n,n+m):
    #     av.append(i)
    val = RSM(art_p,art_BFS)
    if(val[0]==str):
        return val 
    return val[0], val[1][0:n]
    

DEGENERACY = False

# A2 = np.matrix([ [2,1,1,0,0],
#                 [2,3,0,1,0],
#                 [3,1,0,0,1]])
# b2 = np.matrix([18,42,24]).transpose()
# C2 = np.matrix([3,2,0,0,0]).transpose()


# A = np.matrix([ [2,1,1,0,0],
#                 [2,3,0,1,0],
#                 [3,1,0,0,1]])
# b = np.matrix([18,42,24]).transpose()
# C = np.matrix([3,2,0,0,0]).transpose()

A = np.matrix([ [1,0],
                [1,0]])
b = np.matrix([5,4]).transpose()
C = np.matrix([1,0]).transpose()


p = LPP(A,b,C)
z, sol = solve(p)

print("optimal value of objective function is -> ",z)
print("optimal solution is ",sol)


if DEGENERACY:
    print("DEGENERACY found")
