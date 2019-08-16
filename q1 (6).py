'''
Question 1 Skeleton Code
Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import sklearn
import scipy 
from sklearn import metrics

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    x= {}
    for i in range(len(train_data)):
        v = train_data[i]
        if (train_labels[i] not in x):
           x[train_labels[i]] = []
        x[train_labels[i]].append(v)
    means = []
    for key in range(10):
        v = x[key]
        v = np.array(v)
        mean = np.mean(v, axis=0)
        means.append(mean)
    means = np.array(means)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    x= {}
    for i in range(len(train_data)):
        v = train_data[i]
        if (train_labels[i] not in x):
           x[train_labels[i]] = []
        x[train_labels[i]].append(v)
    covariances = np.zeros((10, 64, 64))
    covariances = []
    for key in range(0, 10):
        vc = np.array(x[key])
        vc-= vc.mean(axis=0)
        res = np.dot(np.transpose(vc), np.conjugate(vc)) / (len(vc)-1)
        vc_cov = np.matrix(res)
        # given in question that 0.01I needs to added to each matrix
        Idmatrix = 0.01 * np.identity(vc_cov.shape[0]) 
        vc_f = Idmatrix + vc_cov
        covariances.append(vc_f)
    covariances= np.array(covariances)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    final = []
    v1= (-64/2) * np.log(2 * np.pi)
    for digit in digits:
        res = []
        for i in range(10):
            v2 =  -np.log(np.linalg.det(covariances[i]))/2
            v3 = -((digit - means[i]).T.dot(np.linalg.inv(covariances[i])).dot(digit - means[i]))/2
            vfinal= v1 + v2 + v3
            res.append(vfinal)
        final.append(res)
    final = np.array(final)
    return final


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    res = np.zeros((len(digits), 10))
    lkhd = generative_likelihood(digits, means, covariances)
    for i in range(len(res)):
        for j in range(len(res[0])):
            res[i][j] = lkhd[i][j] - scipy.special.logsumexp(lkhd[i])
    return res

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    res = 0
    for i in range(len(cond_likelihood)):
        res+= cond_likelihood[i][int(labels[i])]
    avg = res/len(digits)
    return avg

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.argmax(cond_likelihood, axis=1)
    pass


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    

   #Evaluation 
   #############Q1 PART A###################
   #Average conditional log-likehood
    
    avg_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print('Average conditional log-likehood on training set:')
    print(avg_train)
    avg_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('Average conditional log-likehood on test set:')
    print(avg_test)
    
    ##### Q1 PART B##########################

    classifydata_train = classify_data(train_data, means, covariances)
    accuracy_train = sklearn.metrics.accuracy_score(train_labels, classifydata_train)
    print('Accuracy on training set:')
    print(accuracy_train)
    classifydata_test = classify_data(test_data, means, covariances)
    accuracy_test = sklearn.metrics.accuracy_score(test_labels, classifydata_test)
    print('Accuracy on test set:')
    print(accuracy_test)


    #########Q1 PART C##########################

    arr1,arr2 = plt.subplots(ncols=10)
    for i in range(10):
        eigenValues, eigenVectors =np.linalg.eig(covariances[i])
        #arr2[i].imshow(np.split(eigenVectors[np.argmax(eigenValues)], 8), cmap='gray')  
        arr2[i].imshow(np.split(eigenVectors[:,i], 8),cmap='gray')  
    plt.show()   
       
if __name__ == '__main__':
    main()
