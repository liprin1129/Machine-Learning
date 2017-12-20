class Update(object):
    @staticmethod
    def gradient_descent(input_x, learning_rate):
        input_x.value -= learning_rate * input_x.gradients[input_x]

    @staticmethod
    def stochastic_gradient_descent(hyper_parameters, learning_rate = 1e-2):
        for parameter in hyper_parameters:
            parameter.value -= learning_rate * parameter.gradients[parameter]
