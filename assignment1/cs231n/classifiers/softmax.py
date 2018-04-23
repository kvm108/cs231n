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

  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(N):
        X_curr = X[i, :]
        scores = X_curr.dot(W)
        f_max = np.max(scores)
        scores = scores - f_max
        
        loss = loss - scores[y[i]] + np.log(np.sum(np.exp(scores)))
        
        for j in range(C):
            dW[:,j] += X_curr*(-1*(j==y[i]) + np.exp(scores[j])/(np.sum(np.exp(scores))))
      
  
  loss/=N
  loss += reg*np.sum(W**2)/2
    
  dW/=N
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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

    
  # loss
  scores = X.dot(W)
  scores -= scores.max()
  scores = np.exp(scores)
  scores_sums = np.sum(scores, axis=1)
  soores_range = scores[range(num_train), y]
  loss = soores_range / scores_sums
  loss = -np.sum(np.log(loss))/num_train + reg * np.sum(W * W)

  # grad
  s = np.divide(scores, scores_sums.reshape(num_train, 1))
  s[range(num_train), y] = - (scores_sums - soores_range) / scores_sums
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W
    
#   # f's and score
#   f = X.dot(W)
#   f_max = np.max(f).reshape(-1, 1)
#   f -= f_max
#   scores = np.exp(f)

#   #loss
#   sum_scores = np.sum(scores, axis =1)
#   scores_range = scores[np.arange(N), y]
#   f_range = f[np.arange(N), y]
#   loss = np.sum(-f_range + np.log(sum_scores))
    
#   #grad
  
#   sum = scores/(sum_scores.reshape(-1, 1))

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

