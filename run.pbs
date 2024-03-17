#!/bin/bash
#PBS -l select=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=RTX6000:cpu_type=rome
#PBS -l walltime=24:00:00
#PBS -N unet

cd $PBS_O_WORKDIR

module purge
module add tools/prod
module add Python/3.11.3-GCCcore-12.3.0
module add virtualenv/20.23.1-GCCcore-12.3.0

source .venv/bin/activate

python3 -m pip install -r requirements.txt

python3 run.py
