import tensorflow as tf

from skluc.utils.datautils import replace_nan


def tf_rbf_kernel(X, Y, gamma):
    """
    Compute the rbf kernel between each entry of X and each line of Y.

    tf_rbf_kernel(x, y, gamma) = exp(- (||x - y||^2 * gamma))

    :param X: A tensor of size n times d
    :param Y: A tensor of size m times d
    :param gamma: The bandwith of the kernel
    :return:
    """
    r1 = tf.reduce_sum(X * X, axis=1)
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.reduce_sum(Y * Y, axis=1)
    r2 = tf.reshape(r2, [1, -1])
    K = tf.matmul(X, tf.transpose(Y))
    K = r1 - 2 * K + r2
    K *= -gamma
    K = tf.exp(K)
    return K


def tf_linear_kernel(X, Y):
    # X = tf.nn.l2_normalize(X, axis=-1)
    # Y = tf.nn.l2_normalize(Y, axis=-1)
    return tf.matmul(X, tf.transpose(Y))

def tf_polynomial_kernel(X, Y, degree=2, gamma=None, **kwargs):
    if gamma is None:
        gamma = tf.div(tf.constant(1, dtype=tf.float32), X.shape[-1].value)
    return tf.pow(tf.add(tf.constant(1, dtype=tf.float32), gamma * tf_linear_kernel(X, Y)), tf.constant(degree, dtype=tf.float32))

def tf_chi_square_PD(X, Y):
    # the drawing of the matrix X expanded looks like a wall
    wall = tf.expand_dims(X, axis=1)
    # the drawing of the matrix Y expanded looks like a floor
    floor = tf.expand_dims(Y, axis=0)
    numerator = 2 * wall * floor
    denominator = wall + floor
    quotient = numerator / denominator
    quotient_without_nan = replace_nan(quotient)
    K = tf.reduce_sum(quotient_without_nan, axis=2)
    return K


def tf_chi_square_CPD(X, Y):
    # the drawing of the matrix X expanded looks like a wall
    wall = tf.expand_dims(X, axis=1)
    # the drawing of the matrix Y expanded looks like a floor
    floor = tf.expand_dims(Y, axis=0)
    numerator = tf.square((wall - floor))
    denominator = wall + floor
    quotient = numerator / denominator
    quotient_without_nan = replace_nan(quotient)
    K = - tf.reduce_sum(quotient_without_nan, axis=2)
    return K


def tf_chi_square_CPD_exp(X, Y, gamma):
    K = tf_chi_square_CPD(X, Y)
    K *= gamma
    return tf.exp(K)


def tf_sigmoid_kernel(X, Y, gamma, constant):
    K = gamma * tf.matmul(X, tf.transpose(Y)) + constant
    return tf.tanh(K)


def tf_laplacian_kernel(X, Y, gamma):
    # the drawing of the matrix X expanded looks like a wall
    wall = tf.expand_dims(X, axis=1)
    # the drawing of the matrix Y expanded looks like a floor
    floor = tf.expand_dims(Y, axis=0)
    D = wall - floor
    D = tf.abs(D)
    D = tf.reduce_sum(D, axis=2)
    K = -gamma * D

    K = tf.exp(K)  # exponentiate K in-place
    return K


def tf_sum_of_kernels(X, Y, kernel_fcts, kernel_params):
    sum_of_kernels = kernel_fcts[0](X, Y, **kernel_params[0])
    i = 1
    while i < len(kernel_fcts):
        k_fct = kernel_fcts[i]
        k_params = kernel_params[i]
        k_result = k_fct(X, Y, **k_params)
        sum_of_kernels = tf.add(sum_of_kernels, k_result)
        i += 1
    return sum_of_kernels


def tf_stack_of_kernels(X, Y, kernel_fcts, kernel_params):
    results_of_kernels = []
    i = 0
    while i < len(kernel_fcts):
        k_fct = kernel_fcts[i]
        k_params = kernel_params[i]
        k_result = k_fct(X, Y, **k_params)
        results_of_kernels.append(k_result)
        i += 1
    # return results_of_kernels

    return tf.concat(results_of_kernels, axis=1, name="Kstack")


if __name__ == '__main__':
    a = tf.Constant(value=0)