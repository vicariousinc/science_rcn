"""
Reference implementation of a two-level RCN model for MNIST classification experiments.

Examples:
- To run a small unit test that trains and tests on 20 images using one CPU 
  (takes ~2 minutes, accuracy is ~60%):
python science_rcn/run.py

- To run a slightly more interesting experiment that trains on 100 images and tests on 20 
  images using multiple CPUs (takes <1 min using 7 CPUs, accuracy is ~90%):
python science_rcn/run.py --train_size 100 --test_size 20 --parallel

- To test on the full 10k MNIST test set, training on 1000 examples 
(could take hours depending on the number of available CPUs, average accuracy is ~97.7+%):
python science_rcn/run.py --full_test_set --train_size 1000 --parallel --pool_shape 25 --perturb_factor 2.0
"""

import argparse
import logging
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from PIL import Image
from matplotlib.pyplot import imread

from science_rcn.inference import test_image
from science_rcn.learning import train_image

LOG = logging.getLogger(__name__)


def run_experiment(
    data_dir="data/MNIST",
    train_size=20,
    test_size=20,
    full_test_set=False,
    pool_shape=(25, 25),
    perturb_factor=2.0,
    parallel=True,
    verbose=False,
    seed=5,
):
    """Run MNIST experiments and evaluate results.

    Parameters
    ----------
    data_dir : string
        Dataset directory.
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    perturb_factor : float
        How much two points are allowed to vary on average given the distance
        between them. See Sec S2.3.2 for details.
    parallel : bool
        Parallelize over multiple CPUs.
    verbose : bool
        Higher verbosity level.
    seed : int
        Random seed used by numpy.random for sampling training set.

    Returns
    -------
    model_factors : ([numpy.ndarray], [numpy.ndarray], [networkx.Graph])
        ([frcs], [edge_factors], [graphs]), outputs of train_image in learning.py.
    test_results : [(int, float)]
        List of (winner_idx, winner_score), outputs of test_image in inference.py.
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    # Multiprocessing set up
    num_workers = None if parallel else 1
    pool = Pool(num_workers)

    train_data, test_data = get_mnist_data_iters(
        data_dir, train_size, test_size, full_test_set, seed=seed
    )

    LOG.info("Training on {} images...".format(len(train_data)))
    train_partial = partial(train_image, perturb_factor=perturb_factor)
    train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(
        9999999
    )
    all_model_factors = list(zip(*train_results))

    LOG.info("Testing on {} images...".format(len(test_data)))
    test_partial = partial(
        test_image, model_factors=all_model_factors, pool_shape=pool_shape
    )
    test_results = pool.map_async(test_partial, [d[0] for d in test_data]).get(9999999)

    # Evaluate result
    correct = 0
    for test_idx, (winner_idx, _) in enumerate(test_results):
        correct += int(test_data[test_idx][1]) == winner_idx // (train_size // 10)
    print("Total test accuracy = {}".format(float(correct) / len(test_results)))

    return all_model_factors, test_results


def get_mnist_data_iters(data_dir, train_size, test_size, full_test_set=False, seed=5):
    """
    Load MNIST data.

    Assumed data directory structure:
        training/
            0/
            1/
            2/
            ...
        testing/
            0/
            ...

    Parameters
    ----------
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    seed : int
        Random seed used by numpy.random for sampling training set.

    Returns
    -------
    train_data, train_data : [(numpy.ndarray, str)]
        Each item reps a data sample (2-tuple of image and label)
        Images are numpy.uint8 type [0,255]
    """
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num_per_class, get_filenames=False):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith("."):
                continue
            if num_per_class is None:
                samples = sorted(os.listdir(cat_path))
            else:
                samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class)

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                # Resize and pad the images to (200, 200)
                # image_arr = imresize(imread(filepath), (112, 112))
                image_arr = np.array(
                    Image.fromarray(imread(filepath)).resize((112, 112))
                )
                img = np.pad(
                    image_arr,
                    pad_width=tuple([(p, p) for p in (44, 44)]),
                    mode="constant",
                    constant_values=0,
                )
                loaded_data.append((img, category))
        return loaded_data

    np.random.seed(seed)
    train_set = _load_data(
        os.path.join(data_dir, "training"), num_per_class=train_size // 10
    )
    test_set = _load_data(
        os.path.join(data_dir, "testing"),
        num_per_class=None if full_test_set else test_size // 10,
    )
    return train_set, test_set


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_size",
        dest="train_size",
        type=int,
        default=20,
        help="Number of training examples.",
    )
    parser.add_argument(
        "--test_size",
        dest="test_size",
        type=int,
        default=20,
        help="Number of testing examples.",
    )
    parser.add_argument(
        "--full_test_set",
        dest="full_test_set",
        action="store_true",
        default=False,
        help="Test on full MNIST test set.",
    )
    parser.add_argument(
        "--pool_shapes",
        dest="pool_shape",
        type=int,
        default=25,
        help="Pool shape.",
    )
    parser.add_argument(
        "--perturb_factor",
        dest="perturb_factor",
        type=float,
        default=2.0,
        help="Perturbation factor.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=5,
        help="Seed for numpy.random to sample training and testing dataset split.",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel",
        default=False,
        action="store_true",
        help="Parallelize over multi-CPUs if True.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Verbosity level.",
    )
    options = parser.parse_args()
    run_experiment(
        train_size=options.train_size,
        test_size=options.test_size,
        full_test_set=options.full_test_set,
        pool_shape=(options.pool_shape, options.pool_shape),
        perturb_factor=options.perturb_factor,
        seed=options.seed,
        verbose=options.verbose,
        parallel=options.parallel,
    )
