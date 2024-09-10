import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss:
    """
    This class combines a softmax activation function and a cross-entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C)) - N is the number of samples, C is the number of classes
        :param target: The target classes (shape: (N,)) - target as class indices (not one-hot encoded)
        :return: A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        # Converts target class indices to one-hot encoding
        """
        target = [0, 2, 1, 0]
        target_one_hot = [ [1, 0, 0]
                           [0, 0, 1]
                           [0, 1, 0]
                           [1, 0, 0] ]
        """
        target_one_hot = np.eye(x.shape[1])[target]

        # Forward pass - Softmax
        """
        Will take the logits (x values) and convert them into probabilities using softmax activation function. 
        Softmax activation function will transform the x value into probabilities of 0 and 1, where the sum of 
        all probabilities class is one. 
        """
        x_softmax = softmax(x)

        # Cross-entropy loss calculation (averaged over samples)
        # Proper numerical stability with epsilon
        """
        Categorical cross-entropy loss function calculation, we added 1e-15 to avoid log(0).
        """
        loss = -np.mean(np.sum(target_one_hot * np.log(x_softmax + 1e-15), axis=1))

        # Backward pass - Simplified gradient for softmax + cross-entropy combined
        input_grad = (x_softmax - target_one_hot) / x.shape[0]

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C)) - N is the number of samples, C is the number of classes
    :return: The softmax of x (same shape as input)
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = np.mean((x-target)**2)
        #Backpropagation - gradient
        input_grad = (2 / x.size) * (x - target)

        return loss, input_grad
