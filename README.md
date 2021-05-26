# Primitive Device Characterization with PySpice

Transistor characterization with
[PySpice](https://pyspice.fabrice-salvaire.fr/). Demonstration downloads and
uses ASU's [PTM](http://ptm.asu.edu/) 90nm devices.

## Quickstart

Adjust the *Setup* section in `predict.hy` and run the script. This will create
an [HDF5](https://www.h5py.org/) with the specified operating point parameters.

```bash
$ hy predict.hy

```

## Installation

Follow the PySpice [installation instructions](https://pyspice.fabrice-salvaire.fr/releases/latest/installation.html) 
and then setup the python environment:

```bash
$ pip install -r requirements.txt
```

## TODO

- [X] Add requirements
- [ ] Add Python Script
- [ ] Add Notebook

