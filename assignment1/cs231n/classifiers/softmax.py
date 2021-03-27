from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    
    for i in range(N):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss += -correct_class_score + np.log(np.sum(np.exp(scores)))
        for j in range(C):
            dW[:, j] += X[i] * np.exp(X[i].dot(W[:, j])) / np.sum(np.exp(X[i].dot(W)))
        dW[:, y[i]] += -X[i]
    
    loss += reg * np.sum(np.square(W))
    loss /= N
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(N), y]
    loss += np.sum(-correct_class_scores + np.log(np.sum(np.exp(scores), axis=1)))
    loss /= N
    loss += reg * np.sum(np.square(W))
    grad_scores = np.ones_like(scores)
    exp_scores = np.exp(scores)
    grad_scores *= exp_scores / np.sum(exp_scores, axis=1).reshape(N, 1)
    grad_scores[np.arange(N), y] -= 1
    dW = np.dot(X.T, grad_scores)
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
