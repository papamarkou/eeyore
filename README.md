![](https://github.com/papamarkou/eeyore/workflows/eeyore/badge.svg)

MCMC methods for neural networks.

eeyore can be installed using pip or anaconda. The anaconda installation does not include ODE modelling functionalilty based on torchdiffeq.

To install eeyore using pip, run
```
pip install eeyore
```

To install eeyore using anaconda, firstly add the required channels by running
```
conda config --add channels pytorch
conda config --add channels conda-forge
```
and subsequently run
```
conda install -c papamarkou eeyore
```
To install eeyore using anaconda without adding any channels, run
```
conda install -c papamarkou -c pytorch -c conda-forge eeyore
```
