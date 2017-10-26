[![](data/vicarious_logo.png)](https://www.vicarious.com)

# Reference implementation of Recursive Cortical Network (RCN)

Reference implementation of a two-level RCN model on MNIST classification. See the *Science* article "A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs" and [Vicarious Blog](https://www.vicarious.com/Common_Sense_Cortex_and_CAPTCHA.html) for details.

> Note: this is an unoptimized reference implementation and is not intended for production.

## Setup

Note: Python 2.7 is supported. The code was tested on OSX 10.11. It may work on other system platforms but not guaranteed.

Before starting please make sure gcc is installed (`brew install gcc`) and up to date in order to compile the various dependencies (particularly numpy).

Clone the repository:

```
git clone https://github.com/vicariousinc/science_rcn.git
```

Simple Install:

```
cd science_rcn
make
```

Manual Install (setting up a virtual environment beforehand is recommended):

```
cd science_rcn
python setup.py install
```

## Run

If you installed via `make` you need to activate the virtual environment:
```
source venv/bin/activate
```

To run a small unit test that trains and tests on 20 MNIST images using one CPU (takes ~2 minutes, accuracy is ~60%):
```
python science_rcn/run.py
```

To run a slightly more interesting experiment that trains on 100 images and tests on 20 MNIST images using multiple CPUs (takes <1 min using 7 CPUs, accuracy is ~90%):
```
python science_rcn/run.py --train_size 100 --test_size 20 --parallel
```

To test on the full 10k MNIST test set, training on 1000 examples (could take hours depending on the number of available CPUs, average accuracy is ~97.7+%):
```
python science_rcn/run.py --full_test_set --train_size 1000 --parallel --pool_shape 25 --perturb_factor 2.0
```

## Blog post

Check out our related [blog post](https://www.vicarious.com/Common_Sense_Cortex_and_CAPTCHA.html).

## Datasets

We used the following datasets for the Science paper:

CAPTCHA datasets

- [reCAPTCHA](http://datasets.vicarious.com/recaptcha.zip) (from [google.com](http://google.com))
- [BotDetect](http://datasets.vicarious.com/botdetect.zip) (from [captcha.com](http://captcha.com))
- [Paypal](http://datasets.vicarious.com/paypal.zip) (from [paypal.com](http://paypal.com))
- [Yahoo](http://datasets.vicarious.com/yahoo.zip) (from [yahoo.com](http://yahoo.com))

MNIST datasets

- Original (available at [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))
- [With occlusions](http://datasets.vicarious.com/mnist-multioccluded.zip) (by us)
- [With noise](http://datasets.vicarious.com/noisyMNIST_tests.zip) (by us)


## MNIST licensing

Yann LeCun (Courant Institute, NYU) and Corinna Cortes (Google Labs, New York) hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.
