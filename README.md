# Project_QTSH
QTSH with multiple states and DVR calculations

Content of the repository:

The root directory contains each model Hamiltonian directory and a Jupyter notebook for plotting.

`Plottting_all.ipynb`, `holstein/`, `superexchange/`, `phenol/`

Run the dynamics calculations from each model directory and do plotting next.

Each model directory contains `DVR` and `TSH` directories as places for running the DVR and TSH calculations. The content is the following.

`DVR`

    `run_all.py` - a script for running multiple DVR calculations with different initial conditions

    `submit.slm` - the SLURM job script used in `run_all.py`

`TSH`

    `run_all.py` - a script for running multiple TSH calculations with different initial conditions and methods

    `submit_TSH.slm` - the SLURM job script used in `run_all.py`

    `recipes/` - a directory contains the recipes of the TSH methods in this work, i.e., FSSH, QTSH, QTSH-SDM, QTSH-XF and QTSH-IDA

