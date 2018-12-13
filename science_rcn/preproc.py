"""A pre-processing layer of the RCN model. See Sec S8.1 for details.
"""
import logging
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve

LOG = logging.getLogger(__name__)


class Preproc(object):
    """
    A simplified preprocessing layer implementing Gabor filters and suppression.

    Parameters
    ----------
    num_orients : int
        Number of edge filter orientations (over 2pi).
    filter_scale : float
        A scale parameter for the filters.
    cross_channel_pooling : bool
        Whether to pool across neighboring orientation channels (cf. Sec S8.1.4).

    Attributes
    ----------
    filters : [numpy.ndarray]
        Kernels for oriented Gabor filters.
    pos_filters : [numpy.ndarray]
        Kernels for oriented Gabor filters with all-positive values.
    suppression_masks : numpy.ndarray
        Masks for oriented non-max suppression.
    """

    def __init__(self,
                 num_orients=16,
                 filter_scale=4.,
                 cross_channel_pooling=False):
        self.num_orients = num_orients
        self.filter_scale = filter_scale
        self.cross_channel_pooling = cross_channel_pooling
        self.suppression_masks = generate_suppression_masks(filter_scale=filter_scale, 
                                                            num_orients=num_orients)

    def fwd_infer(self, img, brightness_diff_threshold=40.):
        """Compute bottom-up (forward) inference.

        Parameters
        ----------
        img : numpy.ndarray
            The input image.
        brightness_diff_threshold : float
            Brightness difference threshold for oriented edges.

        Returns
        -------
        bu_msg : 3D numpy.ndarray of float
            The bottom-up messages from the preprocessing layer. 
            Shape is (num_feats, rows, cols)
        """
        filtered = np.zeros((len(self.filters),) + img.shape, dtype=np.float32)
        for i, kern in enumerate(self.filters):
            filtered[i] = fftconvolve(img, kern, mode='same')
        localized = local_nonmax_suppression(filtered, self.suppression_masks)
        # Threshold and binarize
        localized *= (filtered / brightness_diff_threshold).clip(0, 1)
        localized[localized < 1] = 0

        if self.cross_channel_pooling:
            pooled_channel_weights = [(0, 1), (-1, 1), (1, 1)]
            pooled_channels = [-np.ones_like(sf) for sf in localized]
            for i, pc in enumerate(pooled_channels):
                for channel_offset, factor in pooled_channel_weights:
                    ch = (i + channel_offset) % self.num_orients
                    pos_chan = localized[ch]
                    if factor != 1:
                        pos_chan[pos_chan > 0] *= factor
                    np.maximum(pc, pos_chan, pc)
            bu_msg = np.array(pooled_channels)
        else:
            bu_msg = localized
        # Setting background to -1
        bu_msg[bu_msg == 0] = -1.
        return bu_msg

    @property
    def filters(self):
        return get_gabor_filters(
            filter_scale=self.filter_scale, num_orients=self.num_orients, weights=False)

    @property
    def pos_filters(self):
        return get_gabor_filters(
            filter_scale=self.filter_scale, num_orients=self.num_orients, weights=True)


def get_gabor_filters(size=21, filter_scale=4., num_orients=16, weights=False):
    """Get Gabor filter bank. See Preproc for parameters and returns."""
    def _get_sparse_gaussian():
        """Sparse Gaussian."""
        size = 2 * np.ceil(np.sqrt(2.) * filter_scale) + 1
        alt = np.zeros((int(size), int(size)), np.float32)
        alt[int(size // 2), int(size // 2)] = 1
        gaussian = gaussian_filter(alt, filter_scale / np.sqrt(2.), mode='constant')
        gaussian[gaussian < 0.05 * gaussian.max()] = 0
        return gaussian

    gaussian = _get_sparse_gaussian()
    filts = []
    for angle in np.linspace(0., 2 * np.pi, num_orients, endpoint=False):
        acts = np.zeros((size, size), np.float32)
        x, y = np.cos(angle) * filter_scale, np.sin(angle) * filter_scale
        acts[int(size / 2 + y), int(size / 2 + x)] = 1.
        acts[int(size / 2 - y), int(size / 2 - x)] = -1.
        filt = fftconvolve(acts, gaussian, mode='same')
        filt /= np.abs(filt).sum()  # Normalize to ensure the maximum output is 1
        if weights:
            filt = np.abs(filt)
        filts.append(filt)
    return filts


def generate_suppression_masks(filter_scale=4., num_orients=16):
    """
    Generate the masks for oriented non-max suppression at the given filter_scale.
    See Preproc for parameters and returns.
    """
    size = 2 * int(np.ceil(filter_scale * np.sqrt(2))) + 1
    cx, cy = size // 2, size // 2
    filter_masks = np.zeros((num_orients, size, size), np.float32)
    # Compute for orientations [0, pi), then flip for [pi, 2*pi)
    for i, angle in enumerate(np.linspace(0., np.pi, num_orients // 2, endpoint=False)):
        x, y = np.cos(angle), np.sin(angle)
        for r in range(1, int(np.sqrt(2) * size / 2)):
            dx, dy = round(r * x), round(r * y)
            if abs(dx) > cx or abs(dy) > cy:
                continue
            filter_masks[i, int(cy + dy), int(cx + dx)] = 1
            filter_masks[i, int(cy - dy), int(cx - dx)] = 1
    filter_masks[num_orients // 2:] = filter_masks[:num_orients // 2]
    return filter_masks


def local_nonmax_suppression(filtered, suppression_masks, num_orients=16):
    """
    Apply oriented non-max suppression to the filters, so that only a single 
    orientated edge is active at a pixel. See Preproc for additional parameters.

    Parameters
    ----------
    filtered : numpy.ndarray
        Output of filtering the input image with the filter bank.
        Shape is (num feats, rows, columns).

    Returns
    -------
    localized : numpy.ndarray
        Result of oriented non-max suppression.
    """
    localized = np.zeros_like(filtered)
    cross_orient_max = filtered.max(0)
    filtered[filtered < 0] = 0
    for i, (layer, suppress_mask) in enumerate(list(zip(filtered, suppression_masks))):
        competitor_maxs = maximum_filter(layer, footprint=suppress_mask, mode='nearest')
        localized[i] = competitor_maxs <= layer
    localized[cross_orient_max > filtered] = 0
    return localized
