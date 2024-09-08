from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        #print(f'\nThe parameter value is: {parameter.shape}')
        #print(f'The parameter_grad value is: {parameter_grad.shape}\n')

        updated_parameter = parameter - self.learning_rate * parameter_grad

        return updated_parameter
