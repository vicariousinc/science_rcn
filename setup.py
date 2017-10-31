from __future__ import print_function
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import zipfile


REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def _findRequirements():
    """Read the requirements.txt file and parse into requirements for setup's
    install_requirements option.
    """
    requirements_path = os.path.join(REPO_DIR, 'requirements.txt')
    try:
        return [line.strip()
                for line in open(requirements_path).readlines()
                if not line.startswith('#')]
    except IOError:
        return []

requirements = _findRequirements()

# Check for MNIST data dir
if not os.path.isdir('./data/MNIST'):
    if os.path.exists('./data/MNIST.zip'):
        print("Extracting MNIST data...")
        with zipfile.ZipFile('./data/MNIST.zip', 'r') as z:
            z.extractall('./data/')
    else:
        raise IOError("Cannot find MNIST dataset zip file.")

# Setup C extensions
dilation_module = Extension(
    '_dilation',
    sources=['science_rcn/dilation/dilation.cc'],
)


# Extend the include directories after installing numpy.
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(
    zip_safe=False,
    name='science_rcn',
    version='1.0.0',
    author='Vicarious FPC Inc',
    author_email='opensource@vicarious.com',
    description="RCN reference implementation for '17 Science CAPTCHA paper",
    url='https://github.com/vicariousinc/science_rcn',
    license='MIT',
    cmdclass={'build_ext': build_ext},
    packages=[package for package in find_packages()
              if package.startswith('science_rcn')],
    setup_requires=['numpy>=1.13.3'],
    install_requires=[
        'networkx>=1.11,<1.12',
        'numpy==1.13.3',
        'pillow>=4.1.0,<4.2',
        'scipy>=0.19.0,<0.20',
        'setuptools>=36.5.0'
    ],
    ext_modules=[dilation_module],
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: C'],
    keywords='rcn',
)
