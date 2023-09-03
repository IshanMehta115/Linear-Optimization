import numpy as np
import pandas as pd
def interior_point_method(c,A,b,x0,mu=0.1,tol=0.5,max_iter=500):
    x,y = A.shape;
    n_var = y;m_equal=x;
    x0_prev = x0.reshape(n_var,1);
    z0_prev = mu / x0_prev;
    z0_prev = z0_prev.reshape((n_var,1));
    lambda_par = np.array([0]*m_equal);
    lambda_par = lambda_par.reshape((m_equal,1));
    val_min = 10**9;
    print("shapes x0_prev={} z0_prev={} lambda_par={}".format(x0_prev.shape,z0_prev.shape,lambda_par.shape));
    for iter in range(max_iter):
        alpha = 1/(10);
        X_mat = np.diag(x0_prev.flatten());
        Z_mat = np.diag(z0_prev.flatten());
       
        A_derivative = np.zeros((n_var,m_equal));
        final_mat = np.zeros((n_var+m_equal,n_var+m_equal));
        for i in range(n_var):
            for j in range(m_equal):
                A_derivative[i][j] = A[j][i];
        W_k = np.zeros((n_var,n_var));
        for i in range(n_var):
            W_k[i][i] = 1/(x0_prev[i][0]**2);
        print("shapes X_mat={} Z_mat={} A_derivate={} final_mat={}".format(X_mat.shape,Z_mat.shape,A_derivative.shape,final_mat.shape))
        sigma_mat = np.linalg.inv((np.linalg.inv(X_mat))@Z_mat);
        new_mat= W_k+sigma_mat;
        A_t = A_derivative.T;
        for i in range(n_var):
            for j in range(n_var):
                final_mat[i][j] = new_mat[i][j];
        for i in range(0,n_var):
            for j in range(n_var,n_var+m_equal):
                final_mat[i][j] = A_derivative[i][j-n_var];
        for i in range(n_var,n_var+m_equal):
            for j in range(0,n_var):
                final_mat[i][j] = A_t[i-n_var][j];
        
        inv_mat = np.linalg.inv(final_mat);
        c_val = c;
        A_org = np.array([0.0]*n_var);
        for i in range(m_equal):
            A_org += lambda_par[i]*A[i,:];
        new_mat_rhs = np.array([0.0]*m_equal);
        for i in range(m_equal):
            val = A[i,:].reshape((n_var,1)).T@x0_prev.reshape((n_var,1))+b[i];
            new_mat_rhs[i] = val;
        final_rhs = np.zeros((n_var+m_equal,1));
        for i in range(n_var+m_equal):
            if(i<n_var):
                final_rhs[i][0] = c_val[i]+A_org[i];
            else:
                final_rhs[i][0] = new_mat_rhs[i-n_var];
        result = inv_mat@(-final_rhs);
        result = result.reshape((n_var+m_equal,1));
        dx = np.zeros((n_var,1));
        dy = np.zeros((m_equal,1));
        for i in range(n_var+m_equal):
            if(i<n_var):
                dx[i][0] = result[i][0];
            else:
                dy[i-n_var][0] = result[i][0];
        # dz = np.zeros((n_var,1));
        ones_arr = np.ones((n_var,1));
        for i in range(m_equal):
            z0_prev[i][0] = 0;
        final_arr = (mu*(np.linalg.inv(X_mat)@ones_arr))-(z0_prev.reshape((n_var,1)))-(sigma_mat@dx);
        dz = final_arr.reshape((n_var,1));
        x0_new = x0_prev+alpha*dx;
        lambda_new = lambda_par+alpha*dy;
        z0_new = z0_prev+alpha*dz;
        # print("sum is here",((x0_new-x0_prev)>0).sum());

        # print("x0_new={} lambda0_new={} z0_new={}".format(x0_new,lambda_new,z0_new));
        val_1=0;
        x0_prev = x0_new;
        lambda_par = lambda_new;
        z0_prev = z0_new;
        abb = 0;bcb = 0;
        for i in range(n_var):
            ab = abs(c_val[i]+A_org[i] - 1/x0_prev[i]);
            
            abb = max(ab,abb);
        ones_mat = np.ones((n_var,1)); 
        ones_mat = ones_mat.astype('float');  
        for i in range(m_equal):
            bc = (A[i,:].reshape((n_var,1)).T@x0_prev.reshape((n_var,1))) +b[i];
            bcb = max(abs(bc),bcb);
        X_mat_new = np.diag(x0_new.flatten());
        Z_mat_new = np.diag(z0_new.flatten());
        final_const = np.max(np.abs((X_mat_new@(Z_mat_new@ones_mat)) - mu*(ones_mat)));
    
        
        

        

        if(abb<tol and bcb<tol and abs(final_const)<tol):
            break;
        else:
            print("Still here")
            print(abb,bcb,final_const);
        val_min = min(val_min,10*x0_new[0][0]+9*x0_new[1][0]);
        val_opt = val_min;

        
    print("optimal val={}".format(val_opt));
    print("decision variable values x0_values={}".format(x0_prev));

        # tol constraints;
        




# Example usage

c = np.array([10, 9,0,0])  # Coefficients of the objective function
# 2x1+3x2;
# A = np.array([[1, 1], [2, -1]])  # Coefficients of the equality constraints
A = np.array([[1,2,3,0],[3,2,0,-1]]);b = np.array([-20,-18]);

# b = np.array([3, 2])  # Right-hand side values of the equality constraints

x0 = np.array([6,0.1,4.5667,0.1])  # Initial guess for the primal variables

# Call the interior point method
# x_optimal = interior_point_method(c, A, b, x0,mu=1)

# # Print the optimal solution
# print("Optimal solution: ", x_optimal)

