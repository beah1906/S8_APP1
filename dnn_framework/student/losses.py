import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        soft = softmax(x)
        N = x.shape[0]  # Number of samples
        correct_class_probs = soft[np.arange(N), target]  # Pick the correct class probabilities
        loss = -np.mean(np.log(correct_class_probs))  # Average cross-entropy loss ????????? not sum

        soft[np.arange(N), target] -= 1  # Subtract 1 from the correct class probabilities
        input_grad = soft / N  # Normalize the gradient by the batch size

        return loss, input_grad



def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """

    #return (np.e**x)/np.sum(np.e**x)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


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
        loss = np.sum((x - target) ** 2) / x.size
        input_grad = 2 * (x - target) / x.size
        return loss, input_grad
