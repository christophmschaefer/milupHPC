#!/bin/bash
#PBS -N ParaLoBstar-h5renderer
#PBS -l nodes=1:ppn=20
#PBS -l walltime=00:05:00
#PBS -l mem=4gb
#PBS -q tiny
#PBS -m aeb -M hania.azzam@student.uni-tuebingen.de

source ~/.bashrc

# Loading modules
module load lib/boost/1.69.0
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

# Defining OpenMP number of threads
export OMP_NUM_THREADS=20

# setting memory allocation limit to 200MB per thread
ulimit -s 200000

# debugging output
cd /home/tu/tu_tu/tu_zxogc36/git/miluphpc/H5Renderer/
echo "current directory: $(pwd)"


# executing program
bin/h5renderer 
