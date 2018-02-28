
# coding: utf-8

# In[ ]:

import numpy as np
import sys
import math
from numpy.linalg import inv


# In[ ]:

def calculate_l21_norm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[i,:]||_2
    Input:
    -----
    X: {numpy array}, shape (n_samples, n_features)
    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()
def signedFS(X,**args):
    n_samples, n_features = X.shape
    alpha = 1
    betap = 1
    betan = 100
    gamma = 1000
    lambdaU = 0.5
    lambdaP = 0.5
    lambdaN = 0.5
    # initialize D as identity matrix
    H = np.eye(n_features)
    I = np.eye(n_samples)
    A = AP-AN
    P1 = A
    P2 = np.multiply(O,A**2)
    P = P1 + theta*P2
    L = D - P
    max_iter = 1000
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        eigen_value, eigen_vector = scipy.linalg.eigh(a=P)
        # update M
        M = np.dot(X.transpose(),X) + alpha L + alpha * H
        # partial JU
        JU = np.dot((I - np.dot(np.dot(X,inv(M)),X.transpose()),U)
            +np.dot((I - np.dot(np.dot(X,inv(M)),X.transpose())).transpose(),U)
            +np.dot(betap,np.dot(np.dot(-1,np.multiply(AP,np.multiply(OP,OP))),np.dot(U,VP.transpose()))
            -np.dot((np.multiply(AP,np.multiply(OP,OP))).transpose(),np.dot(U,VP))
            +np.dot(np.multiply(np.dot(np.dot(U,VP),U.transpose()),np.multiply(OP,OP)),np.dot(U,VP.transpose()))
            +np.dot((np.multiply(np.dot(np.dot(U,VP),U.transpose()),np.multiply(OP,OP)))transpose(),np.dot(U,VP)))
            +np.dot(betan,np.dot(np.dot(-1,np.multiply(AN,np.multiply(ON,ON))),np.dot(U,VN.transpose()))
            -np.dot((np.multiply(AN,np.multiply(ON,ON))).transpose(),np.dot(U,VN))
            +np.dot(np.multiply(np.dot(np.dot(U,VN),U.transpose()),np.multiply(ON,ON)),np.dot(U,VN.transpose()))
            +np.dot((np.multiply(np.dot(np.dot(U,VN),U.transpose()),np.multiply(ON,ON)))transpose(),np.dot(U,VN)))
            +gamma*np.dot(L,U)        
        # partial JVP
        JVP = betap * np.dot((np.dot(U.transpose(),np.multiply(np.multiply(OP,OP),np.dot(np.dot(U,VP),U.transpose()),U)
              -np.dot(U.transpose,np.dot(U,np.multiply(np.multiply(OP,OP),AP)))
        # partial JVN
        JVN = betan * np.dot((np.dot(U.transpose(),np.multiply(np.multiply(ON,ON),np.dot(np.dot(U,VN),U.transpose()),U)
              -np.dot(U.transpose,np.dot(U,np.multiply(np.multiply(OP,OP),AN)))
        # update U
        U = U - lambdaU*JU
        # update VP
        VP = VP - lambdaP*JVP                             
        # update VN
        VN = VP - lambdaN*JVN                             
        # update W
        W = np.dot(U,np.dot(numpy.linalg.inv(M),X.transpose))
        # update H
        H = 1/(2*numpy.linalg.norm(W))
        #calculate objective function
        obj[iter_step] = np.linalg.norm((p.dot(X,W)-U)'fro')+np.dot(alpha,calculate_l21_norm(W))
        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break
    return W
        

