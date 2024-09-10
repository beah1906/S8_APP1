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

        target_one_hot = np.eye(x.shape[1])[target]
        loss = -np.mean(np.sum(target_one_hot * np.log(soft + 1e-15), axis=1))
        input_grad = (soft - target_one_hot) / x.shape[0]

        return loss, input_grad



def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """

    #return (np.e**x)/np.sum(np.e**x)
    #return np.exp(x-np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x), axis=1, keepdims=True)
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)
    soft = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return soft

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
        #loss = np.sum((x - target) ** 2) / x.size
        loss = np.mean((x-target)**2)
        #input_grad = 2 * (x - target) / x.size
        input_grad = (2/x.size)*(x-target)
        return loss, input_grad
