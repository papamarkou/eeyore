#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export METAUSER='theodore'
export BASEDIR="/home/$METAUSER"
export BINDIR="$BASEDIR/bin"
export PKGNAME='eeyore'
export PYVERSION='3.6'
export CONDADIR="$BASEDIR/opt/continuum/miniconda/miniconda3"
export CONDABIN="$CONDADIR/bin/conda"
export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'

sudo apt-get update

sudo apt-get install tree

su - $METAUSER -c "wget https://repo.anaconda.com/miniconda/$CONDASCRIPT"
su - $METAUSER -c "chmod u+x $CONDASCRIPT"

su - $METAUSER -c "$SHELL $CONDASCRIPT -b -p $CONDADIR"

su - $METAUSER -c "$CONDABIN create -n $PKGNAME -y -c papamarkou -c pytorch -c conda-forge python=$PYVERSION $PKGNAME"

su - $METAUSER -c "$CONDABIN init $(basename $SHELL)"
su - $METAUSER -c "$CONDABIN config --set auto_activate_base false"

su - $METAUSER -c "mkdir -p $BINDIR"
su - $METAUSER -c "ln -s $CONDABIN $BINDIR"
su - $METAUSER -c "echo \"export PATH=$BINDIR:$PATH\" >> $BASEDIR/.bashrc"

su - $METAUSER -c "rm $CONDASCRIPT"
