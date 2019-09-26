# import tensorflow as tftf
import keras.backend as K
from keras.activations import tanh

from skluc.utils.datautils import replace_nan


def keras_linear_kernel(args, normalize=True, tanh_activation=True):
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    result = K.dot(X, K.transpose(Y))
    if tanh_activation:
        return tanh(result)
    else:
        return result



def keras_chi_square_CPD(args, epsilon=None, tanh_activation=True, normalize=True):
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    # the drawing of the matrix X expanded looks like a wall
    wall = K.expand_dims(X, axis=1)
    # the drawing of the matrix Y expanded looks like a floor
    floor = K.expand_dims(Y, axis=0)
    numerator = K.square((wall - floor))
    denominator = wall + floor
    if epsilon is not None:
        quotient = numerator / (denominator + epsilon)
    else:
        quotient = numerator / denominator
    quotient_without_nan = replace_nan(quotient)
    result = - K.sum(quotient_without_nan, axis=2)
    if tanh_activation:
        return tanh(result)
    else:
        return result


def keras_chi_square_CPD_exp(args, gamma, epsilon=None, tanh_activation=True, normalize=True):
    result = keras_chi_square_CPD(args, epsilon, tanh_activation, normalize)
    result *= gamma
    return K.exp(result)


def keras_rbf_kernel(args, gamma, tanh_activation=True, normalize=True):
    """
    Compute the rbf kernel between each entry of X and each line of Y.

    tf_rbf_kernel(x, y, gamma) = exp(- (||x - y||^2 * gamma))

    :param X: A tensor of size n times d
    :param Y: A tensor of size m times d
    :param gamma: The bandwith of the kernel
    :return:
    """
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    r1 = K.sum(X * X, axis=1)
    r1 = K.reshape(r1, [-1, 1])
    r2 = K.sum(Y * Y, axis=1)
    r2 = K.reshape(r2, [1, -1])
    result = K.dot(X, K.transpose(Y))
    result = r1 - 2 * result + r2
    result *= -gamma
    result = K.exp(result)
    if tanh_activation:
        return tanh(result)
    else:
        return result


def keras_rbf_kernel_auto(args, tanh_activation=True, normalize=True):
    """
    Compute the rbf kernel between each entry of X and each line of Y.

    tf_rbf_kernel(x, y, gamma) = exp(- (||x - y||^2 * gamma))

    :param X: A tensor of size n times d
    :param Y: A tensor of size m times d
    :param gamma: The bandwith of the kernel
    :return:
    """
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    r1 = K.sum(X * X, axis=1)
    r1 = K.reshape(r1, [-1, 1])
    r2 = K.sum(Y * Y, axis=1)
    r2 = K.reshape(r2, [1, -1])
    gamma = K.mean(K.dot(K.transpose(Y), Y))
    result = K.dot(X, K.transpose(Y))
    result = r1 - 2 * result + r2
    result *= -gamma
    result = K.exp(result)
    if tanh_activation:
        return tanh(result)
    else:
        return result


map_kernel_name_function = {
    "linear": keras_linear_kernel,
    "chi2_cpd": keras_chi_square_CPD,
    "chi2_exp_cpd": keras_chi_square_CPD_exp,
    "rbf": keras_rbf_kernel,
    "rbf_auto": keras_rbf_kernel_auto
}