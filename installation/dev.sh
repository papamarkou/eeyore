#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export PKGNAME='eeyore'
export PYVERSION='3.7'
export CONDADIR="$HOME/opt/continuum/miniconda/miniconda3"
export PYPKGDIR="$HOME/opt/python/packages"
export CONDABIN="$CONDADIR/bin/conda"
export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'
export PKGURL="https://github.com/papamarkou/$PKGNAME.git"
export PKGDEVREQS="$PYPKGDIR/$PKGNAME/installation/requirements.txt"

sudo apt-get update

sudo apt-get install tree

wget https://repo.anaconda.com/miniconda/$CONDASCRIPT
chmod u+x $CONDASCRIPT

$SHELL $CONDASCRIPT -b -p $CONDADIR

$CONDABIN create -n $PKGNAME -y python=$PYVERSION

$CONDABIN init $(basename $SHELL)
$CONDABIN config --set auto_activate_base false

source $HOME/.bashrc

mkdir -p $PYPKGDIR
git -C $PYPKGDIR clone $PKGURL
conda activate $PKGNAME
pip install -e $PYPKGDIR/$PKGNAME -r $PKGDEVREQS
conda deactivate

rm $HOME/$CONDASCRIPT
