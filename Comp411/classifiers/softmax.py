import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_examples=X.shape[0]
    num_classess=W.shape[1]
    for i in range(num_examples):
        y_hat= W.T @ X[i,:]
        y_hat-=np.max(y_hat) # for stability reasons
        p=np.exp(y_hat)[y[i]]/np.sum(np.exp(y_hat))
        loss+=-np.log(p)


        X_W_like=np.tile(X[i,:],(num_classess,1)).T
        dW+=(X_W_like * np.exp(y_hat))/np.sum(np.exp(y_hat))
        dW[:,y[i]]+=-X[i,:]

    dW/=num_examples
    loss/=num_examples

    if (regtype=='L2'):
        loss+=reg_l2*np.linalg.norm(W)**2
        dW+=2*reg_l2*W
    else:
        loss+=reg_l2*np.linalg.norm(W)**2+reg_l1*np.linalg.norm(W,ord=1)
        dW+=2*reg_l2*W+reg_l1*np.sign(W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.


    loss = 0.0
    dW = np.zeros_like(W)



    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_examples=X.shape[0]
    num_classess=W.shape[1]

    y_hat= X @ W
    y_hat-=np.max(y_hat,axis=1).reshape(num_examples,1) # for stability reasons
    p=np.exp(y_hat)[tuple([np.arange(num_examples, dtype=int),y])]/np.sum(np.exp(y_hat), axis=1)
    loss=np.sum(-np.log(p))/num_examples
    


    index=np.zeros((num_examples,num_classess))
    index[tuple([np.arange(num_examples, dtype=int),y])]=1
    scores=np.exp(y_hat.T)/np.sum(np.exp(y_hat), axis=1)
    dW+=X.T @ (-index+scores.T)
    dW/=num_examples

    if (regtype=='L2'):
        loss+=reg_l2*np.linalg.norm(W)**2
        dW+=2*reg_l2*W
    else:
        loss+=reg_l2*np.linalg.norm(W)**2+reg_l1*np.linalg.norm(W,ord=1)
        dW+=2*reg_l2*W+reg_l1*np.sign(W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
