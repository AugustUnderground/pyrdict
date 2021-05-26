# Primitive Device Characterization with PySpice

Transistor characterization with
[PySpice](https://pyspice.fabrice-salvaire.fr/). Demonstration downloads and
uses ASU's [PTM](http://ptm.asu.edu/) 90nm devices.

## Quickstart

Adjust the *Setup* section in `predict.hy` or `predict.py` and run the script.
This will create an [HDF5](https://www.h5py.org/) with the specified operating
point parameters.

```bash
$ hy predict.hy     # With hy
$ python predict.py # With Python
$ jupyter lab       # Startup Notebook
```

## Installation

Follow the PySpice [installation instructions](https://pyspice.fabrice-salvaire.fr/releases/latest/installation.html) 
and then setup the python environment:

```bash
$ pip install -r requirements.txt
```

## TODO

- [X] Add requirements
- [X] Add Python Script
- [X] Add Notebook

