## For installation on local machine for development, assuming that conda is available

```
PKGNAME='eeyore'
PKGDIR="${HOME}/opt/python/packages"
PYBIN='python3'

conda update conda
conda update --all

conda create -n ${PKGNAME} python=3.8

conda activate ${PKGNAME}

conda install -c conda-forge numpy
conda install pytorch torchvision torchaudio cpuonly -c pytorch # Linux, Windows
conda install pytorch torchvision torchaudio -c pytorch # Mac
conda install -c conda-forge torchdiffeq
conda install -c papamarkou -c conda-forge kanga

# conda install -c conda-forge spyder

cd ${PKGDIR}
git clone git@github.com:papamarkou/kanga.git
cd kanga
${PYBIN} setup.py develop --user

cd ${PKGDIR}
git clone git@github.com:papamarkou/${PKGNAME}.git
cd ${PKGNAME}
${PYBIN} setup.py develop --user
```
