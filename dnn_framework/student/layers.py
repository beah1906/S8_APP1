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
        self.biases = np.zeros((output_count, 1))

    def get_parameters(self):
        """
        Learnable parameters, i.e. weights and biases
        """
        return {'w': self.weights, 'b': self.biases}

    def get_buffers(self):
        """
        Internal values that are not part of the trainable parameters, i.e. momentum
        """
        return None

    def forward(self, x):
        """
        Forward pass of the network
        """
        y = np.dot(self.weights, x.T) + self.biases
        cache = x

        return y, cache

    def backward(self, output_grad, cache):
        """
        Backward pass of the network
        """
        dl_dx = np.dot(self.weights.T, output_grad)
        dl_dw = np.dot(output_grad, cache)
        dl_db = np.sum(output_grad, axis=1, keepdims=True)

        gradients= {
            'w': dl_dw,
            'b': dl_db
        }

        return dl_dx.T, gradients


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return None

    def get_buffers(self):
        return None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y, y

    def backward(self, output_grad, cache):
        sigmoid_grad = output_grad * ((1 - cache) * cache)

        return sigmoid_grad, cache


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        """
        There are no learnable parameters in ReLU activation function, so simply
        return an empty string.
        """
        return None

    def get_buffers(self):
        """
        There are no trainable data for ReLU activation function, so simply
        return an empty string.
        """
        raise None

    def forward(self, x):
        """
        Apply the forward pass for ReLU activation function.
        Note: This function could be implemented using only:

        output = np.maximum(0, x)

        But not sure if we can use it for this class.
        """
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
                    print(f'The value is : {value}')
                    if value < 0:
                        temp = 0
                        output_temp.append(temp)
                    else:
                        print(f'The output_grad value is: {output_grad[i][j]}')
                        temp = 1 * output_grad[i][j]
                        output_temp.append(temp)
                    print(f'The output value is: {output_temp}')
                output.append(output_temp)

        return np.array(output), cache
