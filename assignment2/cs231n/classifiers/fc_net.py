from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        affine_1, cache_affine_1 = affine_forward(X, self.params['W1'], self.params['b1'])
        relu_1, cache_relu_1 = relu_forward(affine_1)
        affine_2, cache_affine_2 = affine_forward(relu_1, self.params['W2'], self.params['b2'])
        relu_2, cache_relu_2 = relu_forward(affine_2)
        scores = relu_2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        W2 = self.params['W2']
        N = X.shape[0]
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        relu_back_1 = relu_backward(dx, cache_relu_2)
        dX2, dW2, db2 = affine_backward(relu_back_1, cache_affine_2)
        dW2 += self.reg * W2
        relu_back_2 = relu_backward(dX2, cache_relu_1)
        _, dW1, db1 = affine_backward(relu_back_2, cache_affine_1)
        dW1 += self.reg * W1
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        for i in range(self.num_layers):
            if i == 0:
                self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[i]).astype(dtype)
                self.params['b1'] = np.zeros(hidden_dims[i])
                if self.normalization in ('batchnorm', 'layernorm'):
                    self.params['gamma1'] = np.ones(hidden_dims[i])
                    self.params['beta1'] = np.zeros(hidden_dims[i])
            else:
                if i < self.num_layers - 1: # i = 1, 2, ..., L - 2 => layer = 2, 3, ..., L - 1
                    in_dim = hidden_dims[i - 1]
                    out_dim = hidden_dims[i]
                    self.params[f'W{i + 1}'] = weight_scale * np.random.randn(in_dim, out_dim).astype(dtype)
                    self.params[f'b{i + 1}'] = np.zeros(out_dim)
                    if self.normalization in ('batchnorm', 'layernorm'):
                        self.params[f'gamma{i + 1}'] = np.ones(out_dim)
                        self.params[f'beta{i + 1}'] = np.zeros(out_dim)
                else: # layer = L
                    in_dim = hidden_dims[i - 1] 
                    self.params[f'W{i + 1}'] = weight_scale * np.random.randn(in_dim, num_classes).astype(dtype)
                    self.params[f'b{i + 1}'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        regularization_scores = 0.0 # store the regularization 
        cache_affine = {} # store the values of every node performing affine operation
        cache_relu = {} # store the values of every node performing ReLU operation
        cache_normbatch = {}
        cache_layernorm = {}
        cache_dropout = {}
        _X = X.copy() # a copy of X with N * D
        
        # for each hidden layer
        for i in range(self.num_layers):
            layer_weights = self.params[f'W{i + 1}']
            layer_bias = self.params[f'b{i + 1}']
            # add up value of regularization of each layer
            regularization_scores += 0.5 * self.reg * np.sum(np.square(layer_weights))
            if i == self.num_layers - 1: # layer = L
                _X, cache_affine[i + 1] = affine_forward(_X, layer_weights, layer_bias)
            else: # layer = 1, 2, ..., L - 1
                # get the weights and biases of each layer
                layer_weights = self.params[f'W{i + 1}']
                layer_bias = self.params[f'b{i + 1}']
                # forward operations
                _X, cache_affine[i + 1] = affine_forward(_X, layer_weights, layer_bias)
                if self.normalization == "batchnorm":
                    layer_gamma = self.params[f'gamma{i + 1}']
                    layer_beta = self.params[f'beta{i + 1}']
                    _X, cache_normbatch[i + 1] = batchnorm_forward(_X, layer_gamma, layer_beta, self.bn_params[i])
                if self.normalization == "layernorm":
                    layer_gamma = self.params[f'gamma{i + 1}']
                    layer_beta = self.params[f'beta{i + 1}']
                    _X, cache_layernorm[i + 1] = layernorm_forward(_X, layer_gamma, layer_beta, self.bn_params[i])
                _X, cache_relu[i + 1] = relu_forward(_X) 
                if self.use_dropout:
                    _X, cache_dropout[i + 1] = dropout_forward(_X, self.dropout_param)
        scores = _X

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculate loss value and gradient dL/dz 
        loss, dz = softmax_loss(scores, y)
        # add regularization to loss
        loss += regularization_scores
        dX, dW, db = affine_backward(dz, cache_affine[self.num_layers])
        dW += self.reg * self.params['W{0}'.format(self.num_layers)]
        grads[f'W{self.num_layers}'] = dW 
        grads[f'b{self.num_layers}'] = db
        for i in range(1, self.num_layers):
            layer = self.num_layers - i
            if self.use_dropout:
                dropout_back = dropout_backward(dX, cache_dropout[layer])
                relu_back = relu_backward(dropout_back, cache_relu[layer])
            else:
                relu_back = relu_backward(dX, cache_relu[layer])
                
            if self.normalization == 'batchnorm': # batch normalization
                batchnorm_back, dgamma, dbeta = batchnorm_backward(relu_back, cache_normbatch[layer])
                dX, dW, db = affine_backward(batchnorm_back, cache_affine[layer])
                grads[f'gamma{layer}'] = dgamma
                grads[f'beta{layer}'] = dbeta
            elif self.normalization == 'layernorm': # layer normalization
                layernorm_back, dgamma, dbeta = layernorm_backward(relu_back, cache_layernorm[layer])
                dX, dW, db = affine_backward(layernorm_back, cache_affine[layer])
                grads[f'gamma{layer}'] = dgamma
                grads[f'beta{layer}'] = dbeta
            else:
                dX, dW, db = affine_backward(relu_back, cache_affine[layer])
            dW += self.reg * self.params['W{0}'.format(layer)]
            grads[f'W{layer}'] = dW 
            grads[f'b{layer}'] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
