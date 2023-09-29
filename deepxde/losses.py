from . import backend as bkd
from . import config
import torch
from .backend import tf


def mean_absolute_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    return bkd.reduce_mean(bkd.square(y_true - y_pred))


def mean_l2_relative_error(y_true, y_pred):
    return bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))

def l2_error_mask_sigmoid(y_true,y_pred):
    from __main__ import lambdas, sa_lr
    from dde.model import losshistory
    def sigmoid(x):
        y = 1/(1 + bkd.exp(-(x - 2)))
        y[y < 0] = 0 #strictly positive range
        return y
    def sigmoidprime(x):
        y = bkd.exp(-(x-2))/(1 + bkd.exp(-(-x - 2))**2)
        y[y < 0] = 0
        return y #strictly increasing
    gradl = sigmoidprime(lambdas)*losshistory.metrics_test[-1][0]
    lambdas = lambdas + sa_lr*gradl
    return bkd.reduce_mean(sigmoid(lambdas)*(y_true - y_pred))

def uselambdas(y_true,y_pred):
    from __main__ import lambdas
    return bkd.reduce_mean(torch.dot(lambdas,torch.flatten(y_true - y_pred)))

def softmax_cross_entropy(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def zero(*_):
    # TODO: pytorch
    return tf.constant(0, dtype=config.real(tf))


LOSS_DICT = {
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    "mean absolute percentage error": mean_absolute_percentage_error,
    "MAPE": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "mean l2 relative error": mean_l2_relative_error,
    "l2 error mask sigmoid": l2_error_mask_sigmoid,
    "softmax cross entropy": softmax_cross_entropy,
    "use lambdas": uselambdas,
    "zero": zero,
}


def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
