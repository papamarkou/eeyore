from codecs import open
from os import path
from setuptools import find_packages, setup

from eeyore import __version__

url = 'https://github.com/papamarkou/eeyore'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='eeyore',
    version=__version__,
    description='MCMC methods for neural networks',
    long_description=long_description,
    url=url,
    download_url='{0}/archive/v{1}.tar.gz'.format(url, __version__),
    packages=find_packages(),
    license='MIT',
    author='Theodore Papamarkou',
    author_email='theodore.papamarkou@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'
    ],
    keywords=[
        'Bayesian inference',
        'Bayesian neural networks',
        'convergence diagnostics',
        'Markov chain Monte Carlo',
        'posterior predictive distribution'
    ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.19.2', 'torch>=1.9.0', 'kanga>=0.0.20'],
    package_data={'eeyore': ['data/*/x.csv', 'data/*/y.csv', 'data/*/readme.md']},
    include_package_data=True,
    zip_safe=False
)
