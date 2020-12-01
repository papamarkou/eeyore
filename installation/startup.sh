#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export PKGNAME='eeyore'

export CONDADIR="$HOME/opt/continuum/miniconda/miniconda3"

export CONDABIN="$CONDADIR/bin/conda"

export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'

sudo apt-get update

sudo apt-get install tree

wget https://repo.anaconda.com/miniconda/$CONDASCRIPT
chmod u+x $CONDASCRIPT

$SHELL $CONDASCRIPT -b -p $CONDADIR

$CONDABIN create -n $PKGNAME -y -c papamarkou -c pytorch -c conda-forge python=3.8 $PKGNAME

$CONDABIN init $(basename $SHELL)
$CONDABIN config --set auto_activate_base false

rm $CONDASCRIPT
