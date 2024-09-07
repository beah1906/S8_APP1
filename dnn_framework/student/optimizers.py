from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        updated_parameter = parameter - (self.learning_rate * parameter_grad)

        return updated_parameter

    def step(self, parameter_grads):
        """
        Performs an update step for all parameters using their respective gradients.

        Parameters:
        - parameter_grads: Dictionary of gradients for each parameter (same keys as self.parameters).
        """
        for param_name, param_value in self.parameters.items():
            # Get the corresponding gradient
            param_grad = parameter_grads[param_name]

            # Update the parameter using SGD
            self.parameters[param_name] = self._step_parameter(param_value, param_grad, param_name)

