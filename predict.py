import numpy as np
import pandas as pd
import h5py as h5
import multiprocess as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Unit import *

import datetime
import random
import os
import logging
import requests

## Setup file system
lib_path    = 'lib'
model_base  = '90nm_bulk'
model_file  = f'{model_base}.lib' # library has to have '.lib' extension
model_url   = f'http://ptm.asu.edu/modelcard/2006/{model_base}.pm'
device_name = 'nmos'
data_file   = f'{model_base}.h5'
data_path   = 'data'
column_path = 'columns'
pool_size   = 6

## Setup simulation parameters
temperature = 27
VSS         = 0.0
VDD         = 1.2
step_DC     = 0.01
min_VB      = -1.0
step_VB     = -0.1
min_W       = 1e-6
max_W       = 75e-6
num_W       = 10
min_L       = 150e-9
max_L       = 10e-6
num_L       = 10

## Find or download PTM model (http://ptm.asu.edu/)
def setup_library (path, model, url):
    model_path = f'./{path}/{model}'

    if not os.path.isfile(model_path):
        if not os.path.isdir(f'./{path}'):
            os.mkdir(f'./{path}')

        with open(model_path, "wb") as device_model:
            req = requests.get(url, allow_redirects=True)
            device_model.write(req.content)

    return f'./{path}'

## Specify Model Library
lib = SpiceLibrary(setup_library(lib_path, model_file, model_url))

## Create Testbench Circuit
ckt = Circuit('Primitive Device Characterization')

## Include Library
ckt.include(lib[device_name])

## Terminal Voltage Sources
Vd = ckt.V('d', 'D', ckt.gnd, u_V(0))
Vg = ckt.V('g', 'G', ckt.gnd, u_V(0))
Vb = ckt.V('b', 'B', ckt.gnd, u_V(0))

## DUT
M0 = ckt.MOSFET(0, 'D', 'G', ckt.gnd, 'B', model=device_name)

## Save parameters for Database
column_names = [ 'W',   'L' 
               , 'Vds', 'Vgs', 'Vbs', 'vth', 'vdsat'
               , 'id',  'gbs', 'gbd', 'gds', 'gm', 'gmbs'
               , 'cbb', 'csb', 'cdb', 'cgb'
               , 'css', 'csd', 'csg', 'cds' 
               , 'cdd', 'cdg', 'cbs', 'cbd'
               , 'cbg', 'cgd', 'cgs', 'cgg' ]

save_params = [ f'@M0[{p.lower()}]' for p in column_names ]

## Setup simulator
simulator = ckt.simulator( temperature=temperature
                         , nominal_temperature=temperature )

## Save specified parameters
simulator.save_internal_parameters(*save_params)

## Parallelizeable simulation function
def sim_dc(W, L, Vbs):
    M0.w = W
    M0.l = L
    Vb.dc_value = u_V(Vbs)

    analysis = simulator.dc( vd=slice(VSS, VDD, step_DC)
                           , vg=slice(VSS, VDD, step_DC))

    run_data = pd.DataFrame( { p[0]: analysis[p[1]].as_ndarray() 
                               for p in zip(column_names, save_params) } )

    return run_data

## Setup sweep grid
sweep = [ (w,l,vbs) 
          for vbs in np.arange(0.0    , -1.0  , step=-0.1)
          for l in np.linspace(150e-9 , 10e-6 , num=10)
          for w in np.linspace(1e-6   , 75e-6 , num=10) ]

## Run simulation
logging.disable(logging.FATAL)  # Disable logging
with mp.Pool(pool_size) as pool:
    res = tqdm( pool.imap( func=lambda s: sim_dc(*s)
                         , iterable=sweep )
              , total = len(sweep) )
    results = list(res)
logging.disable(logging.NOTSET)

## Concatenate results into one data frame
sim_res = pd.concat(results, ignore_index=True)

## Store data frame to HDF5
with h5.File(data_file, 'w') as h5_file:
    h5_file[data_path] = sim_res.to_numpy()
    h5_file[column_path] = list(sim_res.columns)

## Round terminal voltages for easier filtering
sim_res.Vgs = round(sim_res.Vgs, ndigits=2)
sim_res.Vds = round(sim_res.Vds, ndigits=2)
sim_res.Vbs = round(sim_res.Vbs, ndigits=2)

## Get random traces
traces = sim_res[ (sim_res.Vbs == VSS) 
               & (sim_res.W == random.choice(sim_res.W.unique())) 
               & (sim_res.L == random.choice(sim_res.L.unique())) ]

## Plot some results
fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False)

for v in traces.Vds.unique():
    trace = traces[(traces.Vds == v)]
    ax1.plot(trace.Vgs, trace.id, label = f'Vds = {v} V')
ax1.grid()
ax1.set_yscale('log')
ax1.set_xlabel('Vds [V]')
ax1.set_ylabel('Id [A]')
ax1.legend()
for v in traces.Vgs.unique():
    trace = traces[(traces.Vgs == v)]
    ax2.plot(trace.Vds, trace.id, label = f'Vgs = {v} V')
ax2.grid()
ax2.set_xlabel('Vgs [V]')
ax2.set_ylabel('Id [A]')
ax2.legend()

plt.show()
