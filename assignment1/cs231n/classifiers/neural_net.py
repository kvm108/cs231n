from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    #1st - hidden
    fp_1 = np.dot(X, W1) + b1
    score_1 = np.maximum(0, fp_1)
#     print("hidden layer \n {} \n --------- \n activation \n {} \n".format(fp_1, score_1))
    
    #2nd - output
    scores = np.dot(score_1, W2) + b2
#     print("scores {} \n --------- \n size {} \n".format(scores, scores.shape))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    
#      Karpathy way 
#     print("\n max scores \n {} \n Before \n {} \n ".format(str(np.max(scores)), scores))
    scores -= np.max(scores) #avoid numeric problems
#     print("\n max scores \n {} \n After \n {} \n ".format(np.max(scores), scores))
    
    scores = np.exp(scores)
    scores_sums = np.sum(scores, axis=1, keepdims=True)
#     print('scores sum {} shape \n {}\n '.format(scores_sums, scores_sums.shape))
    
    probs = scores / scores_sums #normalization
#     print('probs shape {} \n probs \n {} \n '.format(probs.shape, probs[range(N), y]))
    
    correct_log_probs = -np.log(probs[range(N), y])
#     print('scores log correct prob {} shape \n {}\n '.format(correct_log_probs, correct_log_probs.shape))
    
    data_loss = np.sum(correct_log_probs)/ N
    reg_loss = reg * (np.sum(W2**2) + np.sum(W1**2)) # why no 0.5?
    
    loss = data_loss + reg_loss
#     print('loss d+r {} \n --------- \n shape {} \n'.format(loss, loss.shape))
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    #gradient on thge scores
    
    dscores = probs
    dscores[range(N), y] -= 1 # since sum of all is 1 because of normalization
    dscores /= N
#     print("After \n dscores {} size {} ".format(dscores, dscores.shape))
     
    # Output -> Hidden
    dW2 = np.dot(score_1.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dW2 += 2 * reg * W2
    
    grads['W2'] = dW2
    grads['b2'] = db2
    
    #bp the grad to hidden layer
    dhidden = np.dot(dscores, W2.T)
    dhidden[score_1==0] = 0 # ReLU - don't bother the ones with -ve loss(which were set to 0)
    
    #Hidden -> Input
    
    #bp to w1 b1
    
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)
    dW1 += 2 * reg * W1
    
    grads['W1'] = dW1
    grads['b1'] = db1
    
#     print("After \n dW2 {} db2 {} ".format(dW2.shape, db2.shape))
#     print("After \n dW1 {} db1 {} ".format(dW1.shape, db1.shape))
    #comment for now
#     s = np.divide(scores, scores_sums.reshape(N, 1))
#     s[range(N), y] = - (scores_sums - correct_s) / scores_sums
#     s /= N
    
#     dW2 = score_1.T.dot(s)
#     db2 = np.sum(s, axis=0)
#     print("dw2 {} db2 {}".format(dW2, db2))
    
#     hidden_j = s.dot(W2.T)
#     hidden_j[score_1 == 0] = 0 #relu
    
#     dW1 = X.T.dot(hidden_j)
#     db1 = np.sum(hidden_j, axis=0)
    
#     grads['W2'] = dW2 + 2 * reg * W2
#     grads['b2'] = db2
#     grads['W1'] = dW1 + 2 * reg * W1
#     grads['b1'] = db1
#-----------------------------------------------------------    
    
#     #calc softmax grad -> w1 value not < 
    
#     binary_scores = np.zeros(scores.shape)
#     binary_scores[np.arange(N), y] = -1
#     final_scores = binary_scores + (scores/scores_sums.reshape(-1,1))
    
    
#     #back prop the grad
    
#     #output
#     # calc grad
#     grads['W2'] = (score_1.T).dot(final_scores/N)
#     grads['W2'] += reg * W2
    
#     #calc bias
#     grads['b2'] = np.sum(final_scores/N, axis = 0)
    
#     #hidden
    
#     dJ_s1 = (final_scores/N).dot(W2.T)
#     dJ_s1[score_1==0] = 0
    
#     # Cacl W1
#     grads['W1'] = (X.T).dot(dJ_s1)
#     grads['W1'] += reg * (W1)
    
#     # Compute b1
#     grads['b1'] = np.sum(dJ_s1, axis=0)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices = np.random.choice(num_train, batch_size)
      X_batch = X[indices]
      y_batch = y[indices] 
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
#       print("Loss history {} ".format(loss))
      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
    
#       print('Shapes \n W1 {} \n------\n b1 {} \n------\n W2 {} \n------\n b2 {}'.format(self.params['W1'].shape, self.params['b1'].shape, self.params['W2'].shape, self.params['b2'].shape))
#       print('\n W1 {} \n------\n b1 {} \n------\n W2 {} \n------\n b2 {}'.format(grads['W1'].shape, grads['b1'].shape, grads['W2'].shape, grads['b2'].shape))
    
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['b1'] -= learning_rate*grads['b1'].ravel()
      self.params['W2'] -= grads['W2']*learning_rate
      self.params['b2'] -= learning_rate*grads['b2'].ravel()
#       print('\n W1 {} \n------\n b1 {} \n------\n W2 {} \n------\n b2 {}'.format(self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']))
#       #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = np.dot(np.maximum(0, np.dot(X, self.params["W1"])+self.params["b1"]), self.params["W2"]) + self.params["b2"]
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


