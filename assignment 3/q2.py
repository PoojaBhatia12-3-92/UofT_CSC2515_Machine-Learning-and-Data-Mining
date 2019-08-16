from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)
import scipy.misc as misc

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

##### Solution Q2 PART B ####################### 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
     

    a_upper= np.exp(np.divide((l2(test_datum , x_train) * -1),((tau ** 2) *2)))
    a_lower= np.exp(misc.logsumexp(np.divide((l2(test_datum, x_train) * -1),((tau ** 2) *2))))
    a=np.divide(a_upper,a_lower)
    a= np.divide(a_upper, a_lower)
    a= np.diag (a[0,:])
    
    t1 = np.dot(np.dot(np.transpose(x_train),a),x_train) 
    t2 = np.dot(np.dot(np.transpose(x_train),a),y_train) 
    
    w_hat= np.linalg.solve(t1+lam * np.eye(t1.shape[0]),t2)
    y_hat= np.dot (test_datum, w_hat)
    y_hat_train= np.dot (x_train,w_hat)
    
    return y_hat, y_hat_train

##### Solution Q2 PART C #######################
def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    
    n = x.shape[0]
    train_x =x[0:int((1.0-val_frac)*n)]
    train_y =y[0:int((1.0-val_frac)*n)]
    
    test_x=x[int((1.0-val_frac)*n):n]
    test_y=y[int((1.0-val_frac)*n):n]
      
    test_n = test_x.shape[0]
    test_losses = np.zeros(taus.shape)
    train_losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        lp=[]
        for i in range(test_n):
            predictions_test, predictions_train = LRLS(test_x[i,:].reshape(1,d),train_x,train_y, tau)
            lp.append(predictions_test)
        lp=np.array(lp)
        test_losses[j] = ((lp-test_y) ** 2).mean()
        train_losses[j] = ((predictions_train -   train_y) ** 2).mean()
    
    return train_losses, test_losses
    
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    #print (train_losses)
    #print (test_losses)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)

