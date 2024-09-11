import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        #Initialize the weights with small random values, to prevent from vanishing gradient or exploding gradient Xavier initializer
        limit = np.sqrt(6 / (input_count + output_count))
        self.weights = np.random.uniform(-limit, limit, (output_count, input_count))
        self.biases = np.zeros((output_count))

    def get_parameters(self):
        """
        Learnable parameters, i.e. weights and biases
        """
        return {'w': self.weights, 'b': self.biases}

    def get_buffers(self):
        """
        Internal values that are not part of the trainable parameters, i.e. momentum
        """
        return {}

    def forward(self, x):
        """
        Forward pass of the network
        """
        # print(f'Forward : Doing the fully connected:')

        dot_product = np.dot(x, self.weights.T)
        y = dot_product + self.biases
        cache = x

        return y, cache

    def backward(self, output_grad, cache):
        """
        Backward pass of the network
        """
        # print(f'Backward : Doing the fully connected:')

        # print(f'The shape for the output_grad is: {output_grad.shape}')
        # print(f'The shape for the cache is: {cache.shape}')
        # print(f'The shape for the weights is: {self.weights.shape}')

        dl_dx = np.dot(output_grad, self.weights)
        # Put the output grad to the original format (We transposed x for the forward pass.)
        dl_dw = np.dot(output_grad.T, cache)
        dl_db = np.sum(output_grad, axis=0, keepdims=True)
        # print(f'dl_db dimensions are: {dl_db.shape}')

        gradients= {
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
        """
        Scale and shift parameters that which are learnable parameters.
        """
        self.gamma = np.ones((input_count,))
        self.beta = np.zeros((input_count,))

        # Buffers for running mean and variance : Inference
        """
        Used to keep track of the mean and variance value across multiples batches 
        when training the neural network. 
        """
        self.running_mean = np.zeros((input_count,))
        self.running_variance = np.ones((input_count,))

        # Momentum used for calculating mean and variance : Inference
        """
        I think it was not mandatory for the class...
        """
        self.alpha = alpha

        # Flag to indicate whether the layer is in training mode or evaluation mode
        self.is_training = True

        # Safety when doing a division to avoid division by 0.
        self.safety = 1e-07

    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def get_buffers(self):
        return {'global_mean': self.running_mean, 'global_variance': self.running_variance}

    def forward(self, x):
        """
        Batch normalization has a different behaviour for training and evaluation, this is why we separate
        the forward into two functions called with a flag.
        """
        if self.is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        """
        Training batch normalization:
        - Use the current batch to calculate the mean and variance.
        - The model is currently being trained which mean that the learnable parameters are changed
        """
        # Put the input data over one axis
        # print(f'Forward : Doing the batch normalization')
        # Compute batch mean and variance
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        # Normalize the batch
        x_normalized = (x - batch_mean) / (np.sqrt(batch_variance) + self.safety)

        # Scale and shift
        y = self.gamma * x_normalized + self.beta

        # Update running mean and variance using momentum
        """
        Keep the running mean and running variance in memory for when we are doing the inference. 
        - When doing the inference we are using this parameters. 
        - In the equation alpha is the momentum (How much of the previous value are retained)
        - The values are dynamically changed during the training to giving more weights to recent values
          while pass values are keep in memory with less importance. 
        """
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
        self.running_variance = self.alpha * self.running_variance + (1 - self.alpha) * batch_variance

        # Cache values for backward pass
        cache = (x, x_normalized, batch_mean, batch_variance)
        return y, cache

    def _forward_evaluation(self, x):
        # Normalize using the running mean and variance
        x_normalized = (x - self.running_mean) / (np.sqrt(self.running_variance) + self.safety)

        # Scale and shift
        y = self.gamma * x_normalized + self.beta

        return y, None

    def backward(self, output_grad, cache):
        # print(f'Backward : Doing the batch normalization')
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
        dmean = np.sum(dx_normalized * -1.0 / np.sqrt(batch_variance), axis=0) + dvariance * np.mean(-2.0 * (x - batch_mean), axis=0)

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
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y, y

    def backward(self, output_grad, cache):
        sigmoid_grad = output_grad * ((1 - cache) * cache)

        return sigmoid_grad, {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        """
        There are no learnable parameters in ReLU activation function, so simply
        return an empty string.
        """
        return {}

    def get_buffers(self):
        """
        There are no trainable data for ReLU activation function, so simply
        return an empty string.
        """
        return {}

    def forward(self, x):
        """
        Apply the forward pass for ReLU activation function.
        Note: This function could be implemented using only:

        output = np.maximum(0, x)

        But not sure if we can use it for this class.
        """
        # print(f'Forward : Doing the ReLU')
        output = []

        if x.ndim == 1:
            # For case when x is a 1D array, i.e. After a fully connected layer
            for value in x:
                if value < 0:
                    output.append(0)
                else:
                    output.append(value)

        elif x.ndim == 2:
            # For case when x is a 2D array, i.e. After a batch normalization layer
            for sample in x:
                output_temp = []
                for value in sample:
                    if value < 0:
                        output_temp.append(0)
                    else:
                        output_temp.append(value)
                output.append(output_temp)

        else:
            raise NotImplementedError()

        return np.array(output), x

    def backward(self, output_grad, cache):
        """
        Apply the backward pass for the ReLU activation function.
        """
        # print(f'Backward : Doing the ReLU')
        output = []

        if cache.ndim == 1:
            # Calculating the backward pass for cache with 1D array
            for i, value in enumerate(cache):
                if value < 0:
                    output.append(0)
                else:
                    output.append(1 * output_grad[i])

        elif cache.ndim == 2:
            # Calculating the backward pass for cache with 2D array
            for i, sample in enumerate(cache):
                output_temp = []
                for j, value in enumerate(sample):
                    if value < 0:
                        temp = 0
                        output_temp.append(temp)
                    else:
                        temp = 1 * output_grad[i][j]
                        output_temp.append(temp)
                output.append(output_temp)

        return np.array(output), {}
