import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    D , C = W.shape 
    score = np.exp(X.dot(W))
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        correct_score = score[i,y[i]]
        sum_score = np.sum(score[i,:])
        loss -= np.log(correct_score/sum_score)
        dW[:,y[i]] -= X[i,:]
        dW += score[i,:]/sum_score*np.tile(X[i,:],(C,1)).T
        
    loss /= num_train
    loss += 0.5*reg*np.sum(W**2)
    
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D , C = W.shape
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    score = np.exp(X.dot(W))
    sum_score = np.sum(score,axis = 1)
    correct_score = score[range(num_train),y]
    loss += np.sum(-1*np.log(correct_score/sum_score))
    
    p_score = score/(sum_score.reshape((-1,1))) 
    p_score[range(num_train),y] -= 1
    dW += X.T.dot(p_score)
    
    loss /= num_train
    loss += 0.5*reg*np.sum(W**2)
    
    dW /= num_train
    dW += reg*W 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

