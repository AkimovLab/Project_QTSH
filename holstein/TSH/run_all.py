
def generate_md_script(_model_indx, _method_indx, _param, _p, _icond_indx, _nsteps):

    if _method_indx == 2:
        name_prefix = F"model{_model_indx}-method{_method_indx}-icond{_icond_indx}-p{_p:.1f}-eps{_param:.1f}"
        name_prefix = name_prefix.replace(".", "_")
    elif _method_indx == 3:
        name_prefix = F"model{_model_indx}-method{_method_indx}-icond{_icond_indx}-p{_p:.1f}-w{_param:.1f}"
        name_prefix = name_prefix.replace(".", "_")
    else:
        name_prefix = F"model{_model_indx}-method{_method_indx}-icond{_icond_indx}-p{_p:.1f}"
        name_prefix = name_prefix.replace(".", "_")

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

from recipes import fssh, qtsh, qtsh_sdm, qtsh_xf, qtsh_ida

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

# nsteps will be updated later
dyn_general = {{ "nsteps":0, "ntraj":2000, "nstates":NSTATES,
                "dt":1.0, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),      
                "mem_output_level":5,
                "properties_to_save":[ "timestep", "time", "q", "p", "Epot_ave", "Ekin_ave", "Etot_ave",
                "states", "se_pop_adi", "se_pop_dia", "sh_pop_adi", "sh_pop_dia_TR", "coherence_adi", "coherence_dia",
                "hvib_adi", "Cadi", "dc1_adi"],
                "prefix":"adiabatic_md", "prefix2":"adiabatic_md"
              }}

dyn_general["nsteps"] = {_nsteps}

#################################
# Give the recipe above an index
method_indx = {_method_indx}
#################################

if method_indx == 0:
    fssh.load(dyn_general)
    
elif method_indx == 1:
    qtsh.load(dyn_general)

elif method_indx == 2:
    qtsh_sdm.load(dyn_general)

elif method_indx == 3:
    qtsh_xf.load(dyn_general)

elif method_indx == 4:
    qtsh_xf.load(dyn_general)
    dyn_general.update({{"use_td_width": 3 }}) # Subotnik width

elif method_indx == 5:
    qtsh_ida.load(dyn_general)

## Initial conditions
icond_nucl = 3 

nucl_params = {{ "ndof":1,
                "q":[-10.0],
                "p":[0.0], 
                "mass":[2000.0], 
                "force_constant":[0.000125], 
                "q_width":[ 0.0 ],
                "p_width":[ 0.0 ],
                "init_type":icond_nucl }}

icond_elec = 0

rep = 1 # adiabatic wfc

istates = []
for i in range(NSTATES):
    istates.append(0.0)        

elec_params = {{"verbosity":2, "init_dm_type":0,
               "ndia":NSTATES, "nadi":NSTATES, 
               "rep":rep, "init_type":icond_elec, "istates":istates
              }}

#####################################
# Select a specific initial condition
icond_indx = {_icond_indx}
#####################################    

if model_indx == 0:
    if icond_indx==0:
        elec_params["istate"] = 0
    elif icond_indx==1:
       elec_params["istate"] = all_model_params[model_indx]["nstates"] - 1

    nucl_params["q"] = [-4.0]
    nucl_params["p"] = [0.0]
    nucl_params["mass"] = [2000]
    nucl_params["force_constant"] = [0.000125]

elif model_indx == 1:
    if icond_indx==0:
        elec_params["istate"] = 0
    elif icond_indx==1:
        elec_params["istate"] = all_model_params[model_indx]["nstates"] - 1

    nucl_params["q"] = [-10.0]
    nucl_params["mass"] = [2000]
    nucl_params["force_constant"] = [0.000125]

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
    
        nucl_params["mass"] = [mu_OH, I]

        nucl_params["q"] =  [r1, 0.0]
        nucl_params["p"] = [15.0, 0.0] # non-zero initial momentum in the r-direction

        q_width = [0.092*units.Angst/math.sqrt(2), 0.55/math.sqrt(2)]
        p_width = [0.5/q_width[0], 0.5/q_width[1]] # Based on the harmonic oscillator WP
        
        elec_params["istate"] = 1

        # force constant calc.
        ks = []
        for dof in range(2):
            ks.append(1. / 4. / nucl_params["mass"][dof] / q_width[dof] ** 4)
    
        nucl_params["force_constant"] = ks

if model_indx == 2:
    nucl_params["p"] = [ {_p}, 0]
else:
    nucl_params["p"] = [ {_p}]

if method_indx == 2:
    eps_param = {_param}
    dyn_general.update({{ "decoherence_eps_param":eps_param }})
if method_indx == 3:
    width = {_param}
    if model_indx == 2:
        WP_W = MATRIX(2,1); WP_W.set(0,0, width); WP_W.set(1,0, q_width[1])
        dyn_general.update({{ "wp_width":WP_W }})
    else:
        WP_W = MATRIX(1,1); WP_W.set(0,0, width)
        dyn_general.update({{ "wp_width":WP_W }})

dyn_params = dict(dyn_general)
dyn_params.update({{ "prefix":"{name_prefix}", 
                    "prefix2":"{name_prefix}" }})

print("Computing " + "{name_prefix}")    

rnd = Random()
res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)


#============ Plotting ==================
pref = "{name_prefix}"

plot_params = {{ "prefix":pref, "filename":"mem_data.hdf", "output_level":4,
                "which_trajectories":[0], "which_dofs":[0], "which_adi_states":list(range(NSTATES)), 
                "which_dia_states":list(range(NSTATES)), 
                "frameon":True, "linewidth":3, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "what_to_plot":["energies", "se_pop_adi", "se_pop_dia", "sh_pop_adi", "sh_pop_dia_TR", "coherence_adi", "coherence_dia" ], 
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":0, "colors":colors
              }}

tsh_dynamics_plot.plot_dynamics(plot_params)"""
    
    return running_script

###

import math
import os
import shutil

model_indx = 0
method_indx = 1
icond_indx = 0

# Momentum Setting
if model_indx == 0:
  ps = [0.0]
elif model_indx == 1:
  incr_p = 0.2
  ps = [3.0 + incr_p*ip for ip in range(11)] + [5.0 + ip for ip in range(1, 16)]
elif model_indx == 2:
  ps = [15.0]

# Decoherence Paramter Setting
param_list = [0.01]

# Time Step Setting
if model_indx == 0:
    Nt = 16001
elif model_indx == 1:
    Nt = 40001
elif model_indx == 2:
    Nt = 4001

base_dir = os.getcwd()

for param in param_list:
    for ip, p in enumerate(ps):
        if method_indx == 2:
            dir_name_suff = F"-p{p:.1f}-eps{param:.1f}"
            dir_name_suff = dir_name_suff.replace(".", "_")
        elif method_indx == 3:
            dir_name_suff = F"-p{p:.1f}-w{param:.1f}"
            dir_name_suff = dir_name_suff.replace(".", "_")
        else:
            dir_name_suff = ""
    
        dir_name = F"RUN{ip}_model{model_indx}_method{method_indx}_icond{icond_indx}" + dir_name_suff
        os.mkdir(dir_name)
        shutil.copy("submit_TSH.slm", dir_name + "/submit_TSH.slm")
        shutil.copytree("recipes", dir_name  + "/recipes")
       
        nsteps = Nt
    
        temp = generate_md_script(model_indx, method_indx, param, p, icond_indx, nsteps)
        with open(dir_name + "/run.py", "w") as fp:
            fp.write(temp)
        
        os.chdir(dir_name)
        os.system("sbatch submit_TSH.slm")
        os.chdir(base_dir)

