#!/bin/bash
#SBATCH --job-name=sans_mcmc           # Name of the job
#SBATCH --output=sans_mcmc_%j.out      # Standard output log (%j will be replaced by the Job ID)
#SBATCH --error=sans_mcmc_%j.err       # Standard error log
#SBATCH --nodes=1                      # Request exactly 1 node
#SBATCH --ntasks=1                     # CRITICAL: Request 1 task (prevents MPI core pinning)
#SBATCH --cpus-per-task=16             # CRITICAL: Allocate 16 cores to that 1 task
#SBATCH --time=48:00:00                # Set a reasonable time limit (hrs:min:sec)

# 1. Set environment variables to prevent C-library thread deadlocks
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# 2. Prevent OpenBLAS from restricting CPU affinity
export OPENBLAS_MAIN_FREE=1


module load mamba

# 3. Activate your Python environment (adjust if using a different environment)
mamba activate /home/esrf/bersenev/cluster_job/Xray

# 4. Execute the Python script using srun to inherit the exact SBATCH allocations
srun python3 cluster_job.py