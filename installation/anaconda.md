## Build, upload to and install from anaconda

### Install prerequisites
```
conda install conda-build
conda install conda-verify
conda install anaconda-client
```

### Add channels and login to anaconda
```
# Add papamarkou, pytorch and conda-forge channels
# This is necessary for the installation not to fail
# See https://github.com/conda/conda-build/issues/3779
conda config --add channels papamarkou
conda config --add channels pytorch
conda config --add channels conda-forge

# Login to anaconda, needed for uploading built packages
anaconda login
```

### Build eeyore using conda
```
cd $HOME
conda skeleton pypi eeyore
conda build --python 3.6 eeyore
```

### Upload to anaconda
```
# anaconda upload $BUILTPKG
```

### Install from anaconda
```
conda install -c papamarkou eeyore
```
