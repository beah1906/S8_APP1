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
        # Convert target to one-hot encoding if it's not already
        target_one_hot = np.eye(x.shape[1])[target]  # Converts target class indices to one-hot encoding

        # Forward pass - Softmax
        x_softmax = softmax(x)

        # Cross-entropy loss calculation (averaged over samples)
        # Proper numerical stability with epsilon
        loss = -np.mean(np.sum(target_one_hot * np.log(x_softmax + 1e-15), axis=1))

        # Backward pass - Simplified gradient for softmax + cross-entropy combined
        input_grad = (x_softmax - target_one_hot) / x.shape[0]

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C)) - N is the number of samples, C is the number of classes
    :return: The softmax of x (same shape as input)
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent numerical instability
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
