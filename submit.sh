#!/bin/bash
#PBS -N PlummerTest
#PBS -l nodes=2:ppn=4:gpus=4:exclusive_process
#PBS -l walltime=00:45:00
#PBS -l pmem=10gb
#PBS -q short
#PBS -m aeb -M hania.azzam@student.uni-tuebingen.de
source ~/.bashrc

# Loading modules
module load compiler/gnu/8.3
module load devel/cuda/10.1
module load mpi/openmpi/4.1-gnu-8.3
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PSM3_MULTI_EP=1

nvidia-smi

# Starting program
#mpirun --bind-to socket --map-by core --report-bindings bin/runner -i 10 -c 0 -f examples/kepler.h5
#mpirun --bind-to core --map-by core --report-bindings bin/runner -i 100 -c 1 -f examples/plummer.h5 -l -L 10
#mpirun --bind-to core --map-by core --report-bindings bin/miluphpc -c 1 -f plummer/plummerN10000seed5.h5 -C testcases/plummer/config.info -l -L 10
mpirun --bind-to core --map-by core --report-bindings bin/runner -f testcases/initial_pd/plN10000seed4056925401.h5 -C testcases/plummer/config.info
#mpirun --bind-to core --map-by core --report-bindings bin/runner -f testcases/initial_pd/sedov_N61.h5 -C testcases/sedov/config.info -m testcases/sedov/material.cfg
