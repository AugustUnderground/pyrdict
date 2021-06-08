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
lib_path      = 'lib'
model_base    = '90nm_bulk'
model_file    = f'{model_base}.lib' # library has to have '.lib' extension
model_url     = f'http://ptm.asu.edu/modelcard/2006/{model_base}.pm'
device_name   = 'nmos'
output_format = 'csv'
pool_size     = 6

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
sim_data = pd.concat(results, ignore_index=True)
columns  = [ 'W','L','Vds','Vgs','Vbs' 
           , 'vth','vdsat','id', 'fug'
           , 'gbs','gbd','gds','gm','gmbs' 
           , 'cgd','cgb','cgs'
           , 'cds','csb','cdb' ]

## Post processing the Data
sim_data['fug'] = sim_data['gm'] / (2 * np.pi * sim_data['cgg'])

cbb,csb,cdb,cgb,\
css,csd,csg,cds,\
cdd,cdg,cbs,cbd,\
cbg,cgd,cgs,cgg = sim_data[ [ 'cbb','csb','cdb','cgb'
                            , 'css','csd','csg','cds'
                            , 'cdd','cdg','cbs','cbd'
                            , 'cbg','cgd','cgs','cgg' ] ].values.T

sim_data['cgd'] = -0.5 * (cdg + cgd)
sim_data['cgb'] = cgg + (0.5 * ( cdg + cgd + csg + cgs ))
sim_data['cgs'] = -0.5 * (cgs + csg)
sim_data['cds'] = -0.5 * (cds + csd)
sim_data['csb'] = css + (0.5 * ( cds + cgs + csd + cgs ))
sim_data['cdb'] = cdd + (0.5 * ( cdg + cds + cgd + csd ))

## Write data frame to disk
if output_format in ['hdf5', 'hdf', 'h5']:
    print(f'Writing data to HDF ...\n')
    with h5.File(data_file, 'w') as h5_file:
        for col in columns:
            h5_file[col] = sim_data[col].to_numpy()
elif output_format == 'csv':
    print(f'Writing data to CSV ...\n')
    sim_data.to_csv(f'{model_base}.csv')
else:
    print(f'No supported file format specified, data won\'t be written.\n')

## Round terminal voltages for easier filtering
sim_data.Vgs = round(sim_data.Vgs, ndigits=2)
sim_data.Vds = round(sim_data.Vds, ndigits=2)
sim_data.Vbs = round(sim_data.Vbs, ndigits=2)

## Get random traces
traces = sim_data[ (sim_data.Vbs == VSS) 
                 & (sim_data.W == random.choice(sim_data.W.unique())) 
                 & (sim_data.L == random.choice(sim_data.L.unique())) ]

## Plot some results
fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False)

for v in np.random.choice(traces.Vds.unique(), 5, replace=False):
    trace = traces[(traces.Vds == v)]
    ax1.plot(trace.Vgs, trace.id, label = f'Vds = {v} V')
ax1.grid()
ax1.set_yscale('log')
ax1.set_xlabel('Vds [V]')
ax1.set_ylabel('Id [A]')
ax1.legend()

for v in np.random.choice(traces.Vgs.unique(), 5, replace=False):
    trace = traces[(traces.Vgs == v)]
    ax2.plot(trace.Vds, trace.id, label = f'Vgs = {v} V')
ax2.grid()
ax2.set_xlabel('Vgs [V]')
ax2.set_ylabel('Id [A]')
ax2.legend()

plt.show()
