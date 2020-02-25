#!/bin/bash
#SBATCH -p all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=bagpipes
#SBATCH --time=48:00:00
#SBATCH --mem=1G
# email notifications (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --mail-type=NONE
#SBATCH --mail-user=adamc@roe.ac.uk

# How to get MPI to work on Cuillin:
# needs to be in place through whole process: module load openmpi/3.1.2/gcc/7.3.0
# compile MultiNest libraries
# install anaconda2 in home directory
# export PATH="/home/adamc/anaconda2/bin:$PATH"
# pip install mpi4py

module purge
cd /home/adamc
source .bash_profile

cd ${SLURM_SUBMIT_DIR}

mpirun -n 32 python fit_legac_v12.py
