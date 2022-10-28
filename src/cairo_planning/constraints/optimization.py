import numpy

def groove_loss(x_val, t, d, c, f, g):
    return -numpy.exp((-(x_val - t)**d) / (2.0 * c**2 ) ) + f * (x_val - t)**g


def groove_loss_derivative(x_val, t, d, c, f, g):
    return -numpy.exp((-(x_val - t)**d) / (2.0 * c**2 ) ) *  ((-d * (x_val - t)) /  (2.0 * c**2)) + g * f * (x_val - t)
