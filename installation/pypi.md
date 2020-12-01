## Upload to and install from pypi and testpypi

### Build eeyore using `python setup.py`
```
python setup.py check
python setup.py sdist bdist_wheel
```

### Upload to pypi and testpypi
```
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

### Install from pypi
```
pip install eeyore
```
