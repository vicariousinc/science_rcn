"""
A reference implementation of max-product loopy belief propagation inference
for a two-level RCN model.

This code is an unoptimized version of what was used to produce the results in the paper.
Note that we use a faster implementation of 2D dilation, instead of the slower
scipy.ndimage.morphology.grey_dilation.
"""
import logging
import numpy as np
import networkx as nx
from numpy.random import rand, randint

from science_rcn.dilation.dilation import dilate_2d
from science_rcn.preproc import Preproc

LOG = logging.getLogger(__name__)


class RCNInferenceError(Exception):
    """Raise for general errors in RCN inference."""
    pass


def test_image(img, model_factors,
               pool_shape=(25, 25), num_candidates=20, n_iters=300, damping=1.0):
    """
    Main function for testing on one image.

    Parameters
    ----------
    img : 2D numpy.ndarray
        The testing image.
    model_factors : ([numpy.ndarray], [numpy.ndarray], [networkx.Graph])
        ([frcs], [edge_factors], [graphs]), output of train_image in learning.py.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    num_candidates : int
        Number of top candidates for backward-pass inference.
    n_iters : int
        Maximum number of loopy BP iterations.
    damping : float
        Damping parameter for loopy BP.

    Returns
    -------
    winner_idx : int
        Training index of the winner feature.
    winner_score : float
        Score of the winning feature.
    """
    # Get bottom-up messages from the pre-processing layer
    preproc_layer = Preproc(cross_channel_pooling=True)
    bu_msg = preproc_layer.fwd_infer(img)

    # Forward pass inference
    fp_scores = np.zeros(len(model_factors[0]))
    for i, (frcs, _, graph) in enumerate(list(zip(*model_factors))):
        fp_scores[i] = forward_pass(frcs,
                                    bu_msg,
                                    graph,
                                    pool_shape)
    top_candidates = np.argsort(fp_scores)[-num_candidates:]

    # Backward pass inference
    winner_idx, winner_score = (-1, -np.inf)  # (training feature idx, score)
    for idx in top_candidates:
        frcs, edge_factors = model_factors[0][idx], model_factors[1][idx]
        rcn_inf = LoopyBPInference(bu_msg, frcs, edge_factors, pool_shape, preproc_layer,
                                   n_iters=n_iters, damping=damping)
        score = rcn_inf.bwd_pass()
        if score >= winner_score:
            winner_idx, winner_score = (idx, score)
    return winner_idx, winner_score


def forward_pass(frcs, bu_msg, graph, pool_shape):
    """
    Forward pass inference using a tree-approximation (cf. Sec S4.2).

    Parameters
    ----------
    frcs : numpy.ndarray of numpy.int
        Nx3 array of (feature idx, row, column), where each row represents a
        single pool center.
    bu_msg : 3D numpy.ndarray of float
        The bottom-up messages from the preprocessing layer.
        Shape is (num_feats, rows, cols)
    graph : networkx.Graph
        An undirected graph whose edges describe the pairwise constraints between
        the pool centers.
        The tightness of the constraint is in the 'perturb_radius' edge attribute.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.

    Returns
    -------
    fp_score : float
        Forward pass score.
    """
    height, width = bu_msg.shape[-2:]
    # Vertical and horizontal pool shapes
    vps, hps = pool_shape

    def _pool_slice(f, r, c):
        assert (r - vps // 2 >= 0 and r + vps - vps // 2 < height and
                c - hps // 2 >= 0 and c + hps - hps // 2 < width), \
            "Some pools are out of the image boundaries. "\
            "Consider increase image padding or reduce pool shapes."
        return np.s_[f,
                     r - vps // 2: r + vps - vps // 2,
                     c - hps // 2: c + hps - hps // 2]

    # Find a schedule to compute the max marginal for the most constrained tree
    tree_schedule = get_tree_schedule(frcs, graph)

    # If we're sending a message out from x to y, it means x has received all
    # incoming messages
    incoming_msgs = {}
    for source, target, perturb_radius in tree_schedule:
        msg_in = bu_msg[_pool_slice(*frcs[source])]
        if source in incoming_msgs:
            msg_in = msg_in + incoming_msgs[source]
            del incoming_msgs[source]
        msg_in = dilate_2d(msg_in, (2 * perturb_radius + 1, 2 * perturb_radius + 1))
        if target in incoming_msgs:
            incoming_msgs[target] += msg_in
        else:
            incoming_msgs[target] = msg_in
    fp_score = np.max(incoming_msgs[tree_schedule[-1, 1]] +
                      bu_msg[_pool_slice(*frcs[tree_schedule[-1, 1]])])
    return fp_score


def get_tree_schedule(frcs, graph):
    """
    Find the most constrained tree in the graph and returns which messages to compute
    it.  This is the minimum spanning tree of the perturb_radius edge attribute.

    See forward_pass for parameters.

    Returns
    -------
    tree_schedules : numpy.ndarray of numpy.int
        Describes how to compute the max marginal for the most constrained tree.
        Nx3 2D array of (source pool_idx, target pool_idx, perturb radius), where
        each row represents a single outgoing factor message computation.
    """
    min_tree = nx.minimum_spanning_tree(graph, 'perturb_radius')
    return np.array([(target, source, graph.edge[source][target]['perturb_radius'])
                     for source, target in nx.dfs_edges(min_tree)])[::-1]


class LoopyBPInference(object):
    """Max-product loopy belief propagation for a two-level RCN model (cf. Sec S4.4).

    Attributes
    ----------
    n_feats, n_rows, n_cols : int, int, int
        Number of features in preprocessing layer, image height, image width.
    n_pools : int
        Number of pools in the model.
    n_factors : int
        Number of edge factors in the model.
    vps, hps : int, int
        Horizontal and vertical pool shape.
    unary_messages : numpy.array
        Unary messages to each variable, obtained by cropping the receptive fields
        from bu_msg. Shape is (n_pools x vps x hps).
    lat_messages : numpy.array
        Lateral message matrix, shape is (n_pools, n_pools, vps, hps). Element
        (v1, v2, r, c) contains the message from v1 to v2, precisely the
        (unnormalized) log-message of pool v2 being in state r, c.
    """

    def __init__(self, bu_msg, frcs, edge_factors, pool_shape, preproc_layer,
                 n_iters=300, damping=1.0, tol=1e-5):
        """
        Parameters
        ----------
        bu_msg : numpy.array of float
            Bottom up messages from preprocessing layer, in the following format:
            (feature idx, row, col).
        frcs : np.ndarray of np.int
            Nx3 array of (feature idx, row, column), where each row represents a
            single pool center.
        edge_factors : numpy.ndarray of numpy.int
            Nx3 array of (source pool index, target pool index, perturb_radius), where
            each row is a pairwise constraints on a pair of pool choices.
        pool_shape : (int, int)
            Vertical and horizontal pool shapes.
        preproc_layer : Preproc
            Pre-processing layer. See preproc.py.
        n_iters : int
            Maximum number of loopy BP iterations.
        damping : float
            Damping parameter for loopy BP.
        tol : float
            Tolerance to determine loopy BP convergence.

        Raises
        ------
        RCNInferenceError
        """
        self.n_feats, self.n_rows, self.n_cols = bu_msg.shape
        self.n_pools, self.n_factors = frcs.shape[0], edge_factors.shape[0]
        self.vps, self.hps = pool_shape
        self.frcs = frcs
        self.bu_msg = bu_msg
        self.edge_factors = edge_factors
        self.preproc_layer = preproc_layer
        self.n_iters = n_iters
        self.damping = damping
        self.tol = tol

        # Check inputs
        if (np.array([0, self.vps // 2, self.hps // 2]) > frcs.min(0)).any():
            raise RCNInferenceError("Some frcs are too small for the provided pool shape")
        if (frcs.max(0) >= np.array([self.n_feats,
                                    self.n_rows - ((self.vps - 1) // 2),
                                    self.n_cols - ((self.hps - 1) // 2)])).any():
            raise RCNInferenceError("Some frcs are too big for the provided pool "
                                    "shape and/or `bu_msg`")
        if (edge_factors[:, :2].min(0) < np.array([0, 0])).any():
            raise RCNInferenceError("Some variable index in `edge_factors` is negative")
        if (edge_factors[:, :2].max(0) >= np.array([self.n_pools, self.n_pools])).any():
            raise RCNInferenceError("Some index in `edge_factors` exceeds the number of vars")
        if (edge_factors[:, 0] == edge_factors[:, 1]).any():
            raise RCNInferenceError("Some factor connects a variable to itself")
        if not issubclass(edge_factors.dtype.type, np.integer):
            raise RCNInferenceError("Factors should be an integer numpy array")

        # Initialize message
        self._reset_messages()
        self.unary_messages = np.zeros((self.n_pools, self.vps, self.hps))
        bu_msg_pert = self.bu_msg + 0.01 * (2 * rand(*bu_msg.shape) - 1)
        for i, (f, r, c) in enumerate(self.frcs):
            rstart = r - self.vps // 2
            cstart = c - self.hps // 2
            self.unary_messages[i] = bu_msg_pert[f,
                                                 rstart:rstart + self.vps,
                                                 cstart:cstart + self.hps]

    def _reset_messages(self):
        """Set all lateral messages to zero."""
        self.lat_messages = np.zeros((2, self.n_factors, self.vps, self.hps))

    @staticmethod
    def compute_1pl_message(in_mess, pert_radius):
        """Compute the outgoing message of a lateral factor given the
        perturbation radius and input message.

        Parameters
        ----------
        in_mess : numpy.array
            Input BP messages to the factor. Each message has shape vps x hps.
        pert_radius : int
            Perturbation radius corresponding to the factor.

        Returns
        -------
        out_mess : numpy.array
            Output BP message (at the opposite end of the factor from the input message).
            Shape is (vps, hps).
        """
        pert_diameter = 2 * pert_radius + 1
        out_mess = dilate_2d(in_mess, (pert_diameter, pert_diameter))
        return out_mess - out_mess.max()

    def new_messages(self):
        """Compute updated set of lateral messages (in both directions).

        Returns
        -------
        new_lat_messages : numpy.array
            Updated set of lateral messages. Shape is (2, n_factors, vps x hps).
        """
        # Compute beliefs
        beliefs = self.unary_messages.copy()
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            beliefs[var_j] += self.lat_messages[0, f]
            beliefs[var_i] += self.lat_messages[1, f]

        # Compute outgoing messages
        new_lat_messages = np.zeros_like(self.lat_messages)
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            new_lat_messages[0, f] = self.compute_1pl_message(
                beliefs[var_i] - self.lat_messages[1, f], pert_radius)
            new_lat_messages[1, f] = self.compute_1pl_message(
                beliefs[var_j] - self.lat_messages[0, f], pert_radius)
        return new_lat_messages

    def bwd_pass(self):
        """Perform max-product loopy BP inference and decode the max-marginals.

        Returns
        -------
        score : float
            The score of the backtraced solution, adjusted for filter overlapping.
        """
        self._reset_messages()
        # Loopy BP with parallel updates
        self.infer_pbp()
        # Decode the max-marginals
        assignments, backtrace_positions, score = self.decode()
        # Check constraints are satisfied
        if not self.laterals_are_satisfied(assignments):
            LOG.debug("Lateral constraints not satisfied. Try increasing the "
                      "number of iterations.")
            score = -np.inf
        return score

    def infer_pbp(self):
        """Parallel loopy BP message passing, modifying state of `lat_messages`.
        See bwd_pass() for parameters.
        """
        for it in range(self.n_iters):
            new_lat_messages = self.new_messages()
            delta = new_lat_messages - self.lat_messages
            self.lat_messages += self.damping * delta
            if np.abs(delta).max() < self.tol:
                LOG.debug("Parallel loopy BP converged in {} iterations".format(it))
                return
        LOG.debug("Parallel loopy BP didn't converge in {} iterations".format(self.n_iters))

    def decode(self):
        """Find pool assignments by decoding the max-marginal messages.

        Returns
        -------
        assignments : 2D numpy.ndarray of int
            Each row is the row and column assignments for each pool.
        backtrace_positions : 3D numpy.ndarray of int
            Sparse top-down activations in the form of (f,r,c).
        score : float
            Sum of log-likelihoods collected by the decoded pool assignments.
        """
        # Compute beliefs
        beliefs = self.unary_messages.copy()
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            beliefs[var_j] += self.lat_messages[0, f]
            beliefs[var_i] += self.lat_messages[1, f]

        assignments = np.zeros((self.n_pools, 2), dtype=np.int)
        backtrace = np.zeros((self.n_feats, self.n_rows, self.n_cols))
        for i, (f, r, c) in enumerate(self.frcs):
            r_max, c_max = np.where(beliefs[i] == beliefs[i].max())
            choice = randint(len(r_max))
            assignments[i] = np.array([r_max[choice], c_max[choice]])
            rstart = r - self.vps // 2
            cstart = c - self.hps // 2
            backtrace[f,
                      rstart + assignments[i, 0],
                      cstart + assignments[i, 1]] = 1
        backtrace_positions = np.transpose(np.nonzero(backtrace))
        score = recount(backtrace_positions, self.bu_msg, self.preproc_layer.pos_filters)
        return assignments, backtrace_positions, score

    def laterals_are_satisfied(self, assignments):
        """Check whether pool assignments satisfy all lateral constraints.

        Parameters
        ----------
        assignments : 2D numpy.ndarray of int
            Row and column assignments for each pool.

        Returns
        -------
        satisfied : bool
            Whether the pool assignments satisfy all lateral constraints.
        """
        satisfied = True
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            rdist, cdist = np.abs(assignments[var_i] - assignments[var_j])
            if not (rdist <= pert_radius and cdist <= pert_radius):
                satisfied = False
                break
        return satisfied


def recount(backtrace_positions, bu_msg, filters):
    """
    Post-processing step to prevent overcounting of log-likelihoods (cf. Sec S8.2).

    Parameters
    ----------
    backtrace_positions : 3D numpy.ndarray of int
        Sparse top-down activations in the format of (f,r,c).
    bu_msg : 3D numpy.ndarray of int
        Bottom-up messages after the pre-processing layer.
    filters : [2D numpy.ndarray of float]
        Filter bank used in the pre-processing layer.

    Returns
    -------
    normalized_score : float
        Score normalized by taking filter overlaps into account.

    Raises
    ------
    RCNInferenceError
    """
    height, width = bu_msg.shape[-2:]
    f_h, f_w = filters[0].shape
    layers = np.zeros((len(backtrace_positions), height, width))
    fo_h, fo_w = f_h // 2, f_w // 2
    from_r, to_r = (np.maximum(0, backtrace_positions[:, 1] - fo_h),
                    np.minimum(height, backtrace_positions[:, 1] - fo_h + f_h))
    from_c, to_c = (np.maximum(0, backtrace_positions[:, 2] - fo_w),
                    np.minimum(width, backtrace_positions[:, 2] - fo_w + f_w))
    from_fr, to_fr = (np.maximum(0, fo_h - backtrace_positions[:, 1]),
                      np.minimum(f_h, height - backtrace_positions[:, 1] + fo_h))
    from_fc, to_fc = (np.maximum(0, fo_w - backtrace_positions[:, 2]),
                      np.minimum(f_w, width - backtrace_positions[:, 2] + fo_w))

    if not np.all(to_r - from_r == to_fr - from_fr):
        raise RCNInferenceError("Numbers of rows of filter and image patches "
                                "({}, {}) do not agree".format(
                                    to_r - from_r, to_fr - from_fr))
    if not np.all(to_c - from_c == to_fc - from_fc):
        raise RCNInferenceError("Numbers of columns of filter and image patches "
                                "({}, {}) do not agree".format(
                                    to_c - from_c, to_fc - from_fc))

    # Normalize activations by taking into account filter overlaps
    weight_sum = np.zeros((height, width))
    for i, (f, r, c) in enumerate(backtrace_positions):
        # Convolve sparse top-down activations with filters
        filt = filters[f][from_fr[i]:to_fr[i], from_fc[i]:to_fc[i]]

        weight_sum[from_r[i]:to_r[i], from_c[i]:to_c[i]] += filt
        layers[i, from_r[i]:to_r[i], from_c[i]:to_c[i]] = \
            filt**2 * bu_msg[f, r, c] / (1e-9 + filt.sum())
    normalized_score = (layers.sum(0) / (1e-9 + weight_sum)).sum()
    return normalized_score
