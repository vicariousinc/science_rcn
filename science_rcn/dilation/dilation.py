"""Fast 2D dilation."""
import numpy as np

from _dilation import max_filter1d, brute_max_filter1d


def dilate_2d(layer, pool_shape, output=None):
    """
    Fast dilation of a feature layer.

    Same as ``scipy.ndimage.morphology.grey_dilation`` but runs much faster.

    Parameters
    ----------
    layer : np.array
        A 2D array to dilate.
    pool_shape : (int, int)
        The dilation shape.

    Returns
    -------
    dilated : np.array
        The dilated output.
    """
    if layer.dtype != np.float32:
        layer = np.array(layer, np.float32)

    dilated = dilate_1d(layer, pool_shape[0], 0, output)
    # Column must come second, as it can operate in place
    dilate_1d(dilated, pool_shape[1], 1, dilated)

    return dilated


def dilate_1d(layer, diameter, axis, output=None):
    if output is None:
        output = np.empty_like(layer)

    if diameter == 1:
        output[:] = layer
    elif diameter < 10:
        brute_max_filter1d(layer, output, diameter, axis)
    else:
        max_filter1d(layer, output, diameter, axis)

    return output
