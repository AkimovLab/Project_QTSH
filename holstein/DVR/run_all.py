
def generate_dvr_script(_model_indx, _p, _icond_indx, _nsteps):

    name_prefix = F"exact-model{_model_indx}-icond{_icond_indx}-p{_p:.1f}".replace(".", "_")
    wfc_prefix = F"wfc{_model_indx}-icond{_icond_indx}-p{_p:.1f}".replace(".", "_")

    running_script = F"""\
#!/usr/bin/env python

import sys
import cmath
import math
import os
import h5py
import matplotlib.pyplot as plt   # plots
import numpy as np
import time
import warnings

from liblibra_core import *
import util.libutil as comn
from libra_py import units
import libra_py.models.Holstein as Holstein
import libra_py.models.Phenol as Phenol

from libra_py import dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
import libra_py.data_savers as data_savers

import libra_py.dynamics.exact.compute as dvr
import libra_py.dynamics.exact.save as dvr_save

colors = {{}}
colors.update({{"11": "#CB3310"}}) #red
colors.update({{"21": "#0077BB"}}) #blue
colors.update({{"31": "#009988"}}) #teal
colors.update({{"41": "#EE7733"}}) #orange
colors.update({{"51": "00FFFF"}}) #cyan
colors.update({{"61": "#EE3377"}}) #magenta
colors.update({{"71": "#AA3377"}}) #purple
colors.update({{"81": "#BBBBBB"}}) #grey

clrs_index = ["11", "21", "31", "41", "51", "61", "71", "81"]

# Define the SE model
class tmp:
    pass

def superexchange(q, params, full_id):

    critical_params = [] 
    default_params = {{ "V_11":0.01, "V_22":0.005, "A_01":0.001, "A_12":0.01, "A_02":0.0  }}
    comn.check_input(params, default_params, critical_params)
    
    V_11 = params["V_11"]
    V_22 = params["V_22"]
    A_01 = params["A_01"]
    A_12 = params["A_12"]
    A_02 = params["A_02"]
    
    n = 3

    Hdia = CMATRIX(n,n)
    Sdia = CMATRIX(n,n)
    d1ham_dia = CMATRIXList();  d1ham_dia.append( CMATRIX(n,n) )
    dc1_dia = CMATRIXList();  dc1_dia.append( CMATRIX(n,n) )
  
    Id = Cpp2Py(full_id)
    indx = Id[-1]

    x = q.col(indx).get(0)
    
    Sdia.identity()    
    
    Hdia.set(0,0,  0 )
    Hdia.set(1,1,  V_11)
    Hdia.set(2,2,  V_22)
    
    exp_factor = math.exp(-0.5 *x * x)
    dexp_factor = -x * math.exp(-0.5 *x * x)

    Hdia.set(0,1,  A_01*exp_factor); Hdia.set(1,0,  A_01*exp_factor)
    Hdia.set(1,2,  A_12*exp_factor); Hdia.set(2,1,  A_12*exp_factor)
        
    d1ham_dia[0].set(0, 1, A_01*dexp_factor); d1ham_dia[0].set(1, 0, A_01*dexp_factor)    
    d1ham_dia[0].set(1, 2, A_12*dexp_factor); d1ham_dia[0].set(2, 1, A_12*dexp_factor)    
                    
    obj = tmp()
    obj.ham_dia = Hdia
    obj.ovlp_dia = Sdia
    obj.d1ham_dia = d1ham_dia
    obj.dc1_dia = dc1_dia

    return obj


def compute_model(q, params, full_id):

    model = params["model"]
    res = None
    
    if model==1:
        res = Holstein.Holstein2(q, params, full_id)
    elif model==2:
        res = superexchange(q, params, full_id)
    elif model==3:
        res = Phenol.Pollien_Arribas_Agostini(q, params, full_id)
    else:
        pass            

    return res

model_params1 = {{"model":1, "model0":1, "nstates":2, "E_n":[0.0, -0.01], "x_n":[0.0,  0.5],"k_n":[0.002, 0.008],"V":0.001}} # holstein
model_params2 = {{"model":2, "model0":2, "nstates":3}} # superexchange
model_params3 = {{"model":3, "model0":3, "nstates":3}} # phenol

all_model_params = [model_params1, model_params2, model_params3]

#################################
# Give the model used an index
model_indx = {_model_indx}
################################

model_params = all_model_params[model_indx]

list_states = [x for x in range(model_params["nstates"])]
NSTATES = model_params["nstates"]

elec_params = {{}}

#####################################
# Select a specific initial condition
icond_indx = {_icond_indx}
#####################################    

if model_indx == 0:
    if icond_indx==0:
        elec_params["istate"] = 0
    elif icond_indx==1:
       elec_params["istate"] = all_model_params[model_indx]["nstates"] - 1

    dt = 1.0
    rmin, rmax = [-25.0], [25.0]
    dx = [0.025]
    x0 = [-4.0]
    p0 = [0.0]
    masses = [2000]
    ks = [0.000125]
    snap_freq = 1000

elif model_indx == 1:
    if icond_indx==0:
        elec_params["istate"] = 0
    elif icond_indx==1:
        elec_params["istate"] = all_model_params[model_indx]["nstates"] - 1

    dt = 1.0
    rmin, rmax = [-150.0], [250.0]
    dx = [0.025]
    x0 = [-10.0]
    p0 = [5.0]
    masses = [2000]
    ks = [0.000125]
    snap_freq = 1000

elif model_indx == 2:
    if icond_indx==0:
        r1 = 0.96944 * units.Angst
        rCC = 1.39403 * units.Angst
        rCH = 1.08441 * units.Angst
        alpha = 109.1333 # degree
    
        mu_OH = 1.57456e-27 / units.m_e
    
        # moments of inertia
        m_H = units.amu
        m_C = 12.0*units.amu
    
        I_1 = mu_OH * (r1 * math.sin(alpha*math.pi/180))**2
        I_2 = 4 * m_C * (rCC * math.sin(math.pi/3))**2 + 4 * m_H * ((rCC + rCH)*math.sin(math.pi/3))**2
        I = 1/(1/I_1 + 1/I_2)
    
        masses = [mu_OH, I]

        x0 = [r1, 0.0]
        p0 = [15.0, 0.0] # non-zero initial momentum in the r-direction

        q_width = [0.092*units.Angst/math.sqrt(2), 0.55/math.sqrt(2)]
        p_width = [0.5/q_width[0], 0.5/q_width[1]] # Based on the harmonic oscillator WP
        
        elec_params["istate"] = 1

        # force constant calc.
        ks = []
        for dof in range(2):
            ks.append(1. / 4. / masses[dof] / q_width[dof] ** 4)

    dt = 10.0
    rmin, rmax = [0.0, -30.0], [60.0, 30.0]
    dx = [0.02, 0.03]
    snap_freq = 10

# For setting the initial state
state_indx = elec_params["istate"]

def potential(q, params):
    full_id = Py2Cpp_int([0,0]) 
    
    return compute_model(q, params, full_id)

pref = F"{name_prefix}"
wfc_pref = F"{wfc_prefix}"

exact_params = {{ "nsteps":{_nsteps}, "dt": dt, "progress_frequency":1/{_nsteps}, "snap_freq":snap_freq,
                  "rmin":rmin, "rmax":rmax, "dx":dx, "nstates":model_params["nstates"],
                  "x0":x0, "p0":p0, "istate":[1,state_indx], "masses":masses, "k":ks,
                  "integrator":"SOFT",
                  "wfc_prefix":wfc_pref, "wfcr_params":[0,0,1], "wfcr_rep":1, 
                  "wfcr_states":[x for x in range(all_model_params[model_indx]["nstates"])],
                  "wfck_params":[0,0,1], "wfck_rep":1, "wfck_states":[x for x in range(model_params["nstates"])],
                  "mem_output_level":0, "txt_output_level":0, "txt2_output_level":0, "hdf5_output_level":3, 
                  "properties_to_save":[ "timestep", "time", "Epot_dia", "Ekin_dia", "Etot_dia",
                                         "Epot_adi", "Ekin_adi", "Etot_adi", "norm_dia", "norm_adi",
                                         "pop_dia", "pop_adi", "q_dia", "q_adi", "p_dia", "p_adi", "coherence_adi"],
                  "prefix":pref, "prefix2":pref,
                  "use_compression":0, "compression_level":[0, 0, 0]
               }}

if model_indx == 2:
    exact_params["p0"] = [ {_p}, 0.0 ]
else:
    exact_params["p0"] = [ {_p} ]

wfc = dvr.init_wfc(exact_params, potential, model_params)
savers = dvr_save.init_tsh_savers(exact_params, model_params, exact_params["nsteps"], wfc)
dvr.run_dynamics(wfc, exact_params, model_params, savers)"""
    
    return running_script

###

import math
import os
import shutil

model_indx = 0
icond_indx = 0

if model_indx == 0:
  ps = [0.0]
elif model_indx == 1:
  incr_p = 0.2
  ps = [3.0 + incr_p*ip for ip in range(11)] + [5.0 + ip for ip in range(1, 16)]
elif model_indx == 2:
  ps = [15.0]

# Time Step Setting
if model_indx == 0:
    Nt = 16001
elif model_indx == 1:
    Nt = 40001
elif model_indx == 2:
    Nt = 401 # dt = 10.0

base_dir = os.getcwd()

for ip, p in enumerate(ps):
    dir_name = F"RUN{ip}_model{model_indx}_icond{icond_indx}" 
    os.mkdir(dir_name)
    shutil.copy("submit.slm", dir_name + "/submit.slm")
    
    nsteps = Nt

    temp = generate_dvr_script(model_indx, p, icond_indx, nsteps)
    with open(dir_name + "/run.py", "w") as fp:
        fp.write(temp)
    
    os.chdir(dir_name)
    os.system("sbatch submit.slm")
    os.chdir(base_dir)

