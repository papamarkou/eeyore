"""eeyore

https://github.com/scidom/eeyore
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='eeyore',
    version='0.0.1',
    description='Monte Carlo methods for neural networks',
    long_description=long_description,
    url='https://github.com/scidom/eeyore',
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
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['Bayesian', 'deep learning', 'Monte Carlo', 'neural networks'],
    package_dir={'': 'src'},
    install_requires=['pytorch',]
)
