from abc import ABC, abstractmethod
import tensorflow as tf


class Optimizer(ABC):
    def __init__(self, lr=0.1):
        self.lr = lr

    @abstractmethod
    def get_update(self, gradients):
        raise NotImplementedError('This method is not implemented '
                                  'for the parent Optimizer class.')

    def apply_gradients(self, gradients, variables):
        update = self.get_update(gradients)
        variables.assign(variables - update)

class SGD(Optimizer):
    def __init__(self, momentum_rate=0.,  **kwargs):
        super().__init__(**kwargs)
        self.update = 0.
        self.momentum_rate = momentum_rate

    def get_update(self, gradients):
        self.update = self.momentum_rate*self.update \
                      + self.lr*gradients
        return self.update


class AdaGrad(Optimizer):
    def __init__(self, var_shape=(31), **kwargs):
        super().__init__(**kwargs)
        self.G_diag = tf.Variable(tf.zeros(var_shape),
                                  trainable=False)

    def get_update(self, gradients):
        self.G_diag.assign(self.G_diag + gradients**2)
        return self.lr*gradients / tf.math.sqrt(self.G_diag)


class RMSProp(Optimizer):
    def __init__(self, forgetting_factor=0.01,
                 var_shape=(31), **kwargs):
        super().__init__(**kwargs)
        self.forgetting_factor = forgetting_factor
        self.running_avg = tf.Variable(tf.zeros(var_shape),
                                       trainable=False)

    def get_update(self, gradients):
        self.running_avg.assign(self.forgetting_factor
                                *self.running_avg
                                + (1-self.forgetting_factor)
                                *gradients**2)
        return self.lr*gradients / tf.math.sqrt(self.running_avg)


class ADAM(Optimizer):
    def __init__(self, beta_1 = 0.9, beta_2 = 0.999,
                 var_shape=(31), epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.running_avg = tf.Variable(tf.zeros(var_shape),
                                       trainable=False)
        self.running_var = tf.Variable(tf.zeros(var_shape),
                                       trainable=False)
        self.training_iter = 1

    def get_update(self, gradients):
        self.running_avg.assign(self.beta_1*self.running_avg
                                + (1-self.beta_1)*gradients)
        self.running_var.assign(self.beta_2*self.running_var
                                + (1-self.beta_2)*gradients**2)
        running_avg_hat = self.running_avg \
                          / (1-self.beta_1**self.training_iter)
        running_var_hat = self.running_var\
                          / (1-self.beta_2**self.training_iter)
        self.training_iter += 1
        return self.lr * running_avg_hat \
               / (tf.math.sqrt(running_var_hat) + self.epsilon)








