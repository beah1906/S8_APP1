# noinspection PyUnresolvedReferences
import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        """
        Initializes the layer with input and output dimensions.
        :param input_count: Number of input features
        :param output_count: Number of output features
        """
        limit = np.sqrt(6 / (input_count + output_count))
        self.w = np.random.uniform(-limit, limit, (output_count, input_count))
        #self.biases = np.zeros((output_count, 1))
        # Initialize weights (w) and biases (b) randomly
        #self.w = np.random.randn(output_count, input_count) * 0.01  # Weight initialization with small values
        self.b = np.zeros((output_count,))  # Bias initialization with zeros


    def get_parameters(self):
        """
        Returns the learnable parameters (weights and biases).
        :return: Dictionary containing weights 'w' and biases 'b'
        """
        return {'w': self.w, 'b': self.b}

    def get_buffers(self):
        """
        No internal buffers for this layer.
        """
        return {}

    def forward(self, x):
        """
        Forward pass of the fully connected layer.
        :param x: Input tensor (batch_size, input_count)
        :return: Output tensor (batch_size, output_count) and cache for backward pass
        """
        # Linear transformation: y = x * w^T + b
        y = np.dot(x, self.w.T) + self.b

        # Cache input x for backward pass
        cache = x
        return y, cache

    def backward(self, output_grad, cache):
        """
        Backward pass of the network
        """
        x = cache
        dl_dx = np.dot(output_grad, self.w)
        dl_dw = np.dot(output_grad.T, x)
        dl_db = np.sum(output_grad, axis=0, keepdims=True)

        gradients = {
            'w': dl_dw,
            'b': dl_db
        }

        return dl_dx, gradients


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        # Learnable parameters:
        self.gamma = np.ones((input_count,))
        self.beta = np.zeros((input_count,))

        # Buffers for running mean and variance : Inference
        self.running_mean = np.zeros((input_count,))
        self.running_variance = np.ones((input_count,))

        # Momentum used for calculating mean and variance : Inference
        self.alpha = alpha

        # Flag to indicate whether the layer is in training mode or evaluation mode
        self.is_training = True

    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def get_buffers(self):
        return {'global_mean': self.running_mean, 'global_variance': self.running_variance}

    def forward(self, x):
        if self.is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        # Compute batch mean and variance
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        # Normalize the batch
        x_normalized = (x - batch_mean) / np.sqrt(batch_variance)

        # Scale and shift
        y = self.gamma * x_normalized + self.beta

        # Update running mean and variance using momentum
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
        self.running_variance = self.alpha * self.running_variance + (1 - self.alpha) * batch_variance

        # Cache values for backward pass
        cache = (x, x_normalized, batch_mean, batch_variance)
        return y, cache

    def _forward_evaluation(self, x):
        # Normalize using the running mean and variance
        x_normalized = (x - self.running_mean) / np.sqrt(self.running_variance + 1e-5)

        # Scale and shift
        y = self.gamma * x_normalized + self.beta

        return y, None

    def backward(self, output_grad, cache):
        x, x_normalized, batch_mean, batch_variance = cache

        N = x.shape[0]

        # Gradient wrt beta and gamma
        dgamma = np.sum(output_grad * x_normalized, axis=0)
        dbeta = np.sum(output_grad, axis=0)

        # Gradient wrt normalized input
        dx_normalized = output_grad * self.gamma

        # Gradient wrt variance
        dvariance = np.sum(dx_normalized * (x - batch_mean) * -0.5 * np.power(batch_variance, -1.5), axis=0)

        # Gradient wrt mean
        dmean = np.sum(dx_normalized * -1.0 / np.sqrt(batch_variance), axis=0) + dvariance * np.mean(
            -2.0 * (x - batch_mean), axis=0)

        # Gradient wrt input
        dx = (dx_normalized / np.sqrt(batch_variance)) + (dvariance * 2.0 * (x - batch_mean) / N) + (dmean / N)

        return dx, {'gamma': dgamma, 'beta': dbeta}

    def train(self):
        """Set the layer to training mode."""
        self.is_training = True

    def eval(self):
        """Set the layer to evaluation mode."""
        self.is_training = False







class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        sigma = 1/(1+np.e**(-x))
        cache = sigma
        return 1/(1+np.e**(-x)), cache

    def backward(self, output_grad, cache):
        sigma = cache
        input_grad = output_grad * sigma * (1- sigma)
        return input_grad, cache


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        relu = np.maximum(0, x)
        cache = x
        return relu, cache


    def backward(self, output_grad, cache):
        input_grad = output_grad * (cache > 0)
        return input_grad, {}
