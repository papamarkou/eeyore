#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export METAUSER='theodore'
export BASEDIR="/home/$METAUSER"
export PKGNAME='eeyore'
export PYVERSION='3.6'
export CONDADIR="$HOME/opt/continuum/miniconda/miniconda3"
export PYPKGDIR="$HOME/opt/python/packages"
export CONDABIN="$CONDADIR/bin/conda"
export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'
export PKGURL="https://github.com/papamarkou/$PKGNAME.git"

sudo apt-get update

sudo apt-get install tree

su - $METAUSER -c "wget https://repo.anaconda.com/miniconda/$CONDASCRIPT"
su - $METAUSER -c "chmod u+x $CONDASCRIPT"

su - $METAUSER -c "$SHELL $CONDASCRIPT -b -p $CONDADIR"

su - $METAUSER -c "$CONDABIN create -n $PKGNAME -y python=$PYVERSION"

su - $METAUSER -c "$CONDABIN init $(basename $SHELL)"
su - $METAUSER -c "$CONDABIN config --set auto_activate_base false"

su - $METAUSER -c "mkdir -p $PYPKGDIR;
cd $PYPKGDIR;
git clone $PKGURL;
cd $PKGNAME;
$CONDABIN activate $PKGNAME;
python setup.py develop --user"

rm $CONDASCRIPT
