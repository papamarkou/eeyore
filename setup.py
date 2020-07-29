"""eeyore

https://github.com/papamarkou/eeyore
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='eeyore',
    version='0.0.2',
    description='MCMC methods for neural networks',
    long_description=long_description,
    url='https://github.com/papamarkou/eeyore',
    author='Theodore Papamarkou',
    author_email='papamarkout@ornl.gov',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    keywords=['Bayesian', 'deep learning', 'Markov chains', 'MCMC', 'Monte Carlo', 'neural networks'],
    package_dir={'eeyore': 'eeyore'},
    install_requires=['numpy', 'torch>=1.3.0', 'torchdiffeq'],
    dependency_links=['git+https://github.com/rtqichen/torchdiffeq.git#egg=torchdiffeq'],
    package_data={'eeyore': ['data/*/x.csv', 'data/*/y.csv', 'data/*/readme.md']},
    include_package_data=True,
    zip_safe=False
)
