"""
Learn a two-layer RCN model. See train_image for the main entry.
"""
from collections import namedtuple
import logging
import numpy as np
import networkx as nx
from scipy.spatial import distance, cKDTree

from science_rcn.preproc import Preproc

LOG = logging.getLogger(__name__)

ModelFactors = namedtuple("ModelFactors", "frcs edge_factors graph")


def train_image(img, perturb_factor=2.0, num_iterations=100):
    """Main function for training on one image.

    Parameters
    ----------
    num_iterations
    img : 2D numpy.ndarray
        The training image.
    perturb_factor : float
        How much two points are allowed to vary on average given the distance
        between them. See Sec S2.3.2 for details.

    Returns
    -------
    frcs : numpy.ndarray of numpy.int
        Nx3 array of (feature idx, row, column), where each row represents a
        single pool center
    edge_factors : numpy.ndarray of numpy.int
        Nx3 array of (source pool index, target pool index, perturb_radius), where
        each row is a pairwise constraints on a pair of pool choices.
    graph : networkx.Graph
        An undirected graph whose edges describe the pairwise constraints between
        the pool centers.
        The tightness of the constraint is in the 'perturb_radius' edge attribute.
    """
    # Pre-processing layer (cf. Sec 4.2.1)
    preproc_layer = Preproc()
    bu_msg = preproc_layer.fwd_infer(img)
    # Sparsification (cf. Sec 5.1.1)
    frcs = sparsify(bu_msg)

    # Training loop for specified number of iterations
    for iteration in range(num_iterations):
        # Lateral learning (cf. 5.2)
        graph, edge_factors = learn_laterals(frcs, bu_msg, perturb_factor=perturb_factor)

    return ModelFactors(frcs, edge_factors, graph)


def sparsify(bu_msg, suppress_radius=3, activation_threshold=0.5):
    """Make a sparse representation of the edges by greedily selecting features from the
    output of preprocessing layer and suppressing overlapping activations.

    Parameters
    ----------
    bu_msg : 3D numpy.ndarray of float
        The bottom-up messages from the preprocessing layer.
        Shape is (num_feats, rows, cols)
    suppress_radius : int
        How many pixels in each direction we assume this filter
        explains when included in the sparsification.
    activation_threshold : float
        The minimum activation value a pixel must have to be considered as a feature.

    Returns
    -------
    frcs : numpy.ndarray
        Selected features along with their locations.
    """
    frcs = []
    img = bu_msg.max(0) > activation_threshold
    while True:
        r, c = np.unravel_index(img.argmax(), img.shape)
        if not img[r, c]:
            break
        frcs.append((bu_msg[:, r, c].argmax(), r, c))
        img[
            r - suppress_radius: r + suppress_radius + 1,
            c - suppress_radius: c + suppress_radius + 1,
        ] = False
    return np.array(frcs)


def learn_laterals(frcs, bu_msg, perturb_factor, use_adjaceny_graph=False):
    """Given the sparse representation of each training example,
    learn perturbation laterals. See train_image for parameters and returns.
    """
    if use_adjaceny_graph:
        graph = make_adjacency_graph(frcs, bu_msg)
        graph = adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(list(range(frcs.shape[0])))

    graph = add_underconstraint_edges(frcs, graph, perturb_factor=perturb_factor)
    graph = adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)

    edge_factors = np.array(
        [
            (edge_source, edge_target, edge_attrs["perturb_radius"])
            for edge_source, edge_target, edge_attrs in graph.edges(data=True)
        ]
    )
    return graph, edge_factors


def make_adjacency_graph(frcs, bu_msg, max_dist=3):
    """Make a graph based on contour adjacency."""
    preproc_pos = np.transpose(np.nonzero(bu_msg > 0))[:, 1:]
    preproc_tree = cKDTree(preproc_pos)
    # Assign each preproc to the closest F1
    f1_bus_tree = cKDTree(frcs[:, 1:])
    _, preproc_to_f1 = f1_bus_tree.query(preproc_pos, k=1)
    # Add edges
    preproc_pairs = np.array(list(preproc_tree.query_pairs(r=max_dist, p=1)))
    f1_edges = np.array(
        list({(x, y) for x, y in preproc_to_f1[preproc_pairs] if x != y})
    )

    graph = nx.Graph()
    graph.add_nodes_from(list(range(frcs.shape[0])))
    graph.add_edges_from(f1_edges)
    return graph


def add_underconstraint_edges(
        frcs, graph, perturb_factor=2.0, max_cxn_length=100, tolerance=4
):
    """Examines all pairs of variables and greedily adds pairwise constraints
    until the pool flexibility matches the desired amount of flexibility specified by
    perturb_factor and tolerance.

    Parameters
    ----------
    frcs : numpy.ndarray of numpy.int
        Nx3 array of (feature idx, row, column), where each row represents a
        single pool center.
    perturb_factor : float
        How much two points are allowed to vary on average given the distance
        between them.
    max_cxn_length : int
        The maximum radius to consider adding laterals.
    tolerance : float
        How much relative error to tolerate in how much two points vary relative to each
        other.

    Returns
    -------
    graph : see train_image.
    """
    graph = graph.copy()
    f1_bus_tree = cKDTree(frcs[:, 1:])

    close_pairs = np.array(list(f1_bus_tree.query_pairs(r=max_cxn_length)))
    dists = [distance.euclidean(frcs[x, 1:], frcs[y, 1:]) for x, y in close_pairs]

    for close_pairs_idx in np.argsort(dists):
        source, target = close_pairs[close_pairs_idx]
        dist = dists[close_pairs_idx]

        try:
            perturb_dist = nx.shortest_path_length(
                graph, source, target, "perturb_radius"
            )
        except nx.NetworkXNoPath:
            perturb_dist = np.inf

        target_perturb_dist = dist / float(perturb_factor)
        actual_perturb_dist = max(0, np.ceil(target_perturb_dist))
        if perturb_dist >= target_perturb_dist * tolerance:
            graph.add_edge(source, target, perturb_radius=int(actual_perturb_dist))
    return graph


def adjust_edge_perturb_radii(frcs, graph, perturb_factor=2):
    """Returns a new graph where the 'perturb_radius' has been adjusted to account for
    rounding errors. See train_image for parameters and returns.
    """
    graph = graph.copy()

    total_rounding_error = 0
    for n1, n2 in nx.edge_dfs(graph):
        desired_radius = distance.euclidean(frcs[n1, 1:], frcs[n2, 1:]) / perturb_factor

        upper = int(np.ceil(desired_radius))
        lower = int(np.floor(desired_radius))
        round_up_error = total_rounding_error + upper - desired_radius
        round_down_error = total_rounding_error + lower - desired_radius
        if abs(round_up_error) < abs(round_down_error):
            graph.edges[n1, n2]["perturb_radius"] = upper
            total_rounding_error = round_up_error
        else:
            graph.edges[n1, n2]["perturb_radius"] = lower
            total_rounding_error = round_down_error
    return graph
