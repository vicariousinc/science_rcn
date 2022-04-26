import os
import setuptools
import zipfile

NAME = "science_rcn"
VERSION = "1.0.1"

# Check for MNIST data dir
if not os.path.isdir("./data/MNIST"):
    if os.path.exists("./data/MNIST.zip"):
        print("Extracting MNIST data...")
        with zipfile.ZipFile("./data/MNIST.zip", "r") as z:
            z.extractall("./data/")
    else:
        raise IOError("Cannot find MNIST dataset zip file.")


if __name__ == "__main__":
    setuptools.setup(
        name=NAME,
        version=VERSION,
        author="Vicarious FPC Inc",
        author_email="opensource@vicarious.com",
        description="RCN reference implementation for '17 Science CAPTCHA paper",
        url="https://github.com/vicariousinc/science_rcn",
        license="MIT",
        packages=setuptools.find_namespace_packages("science_rcn"),
    )
