#!/bin/bash
#SBATCH --job-name=sans_Brush_micelles_MCMC       # Name of the job
#SBATCH --output=sans_Brush_micelles_MCMC_%j.out  # Standard output log (%j = job ID)
#SBATCH --error=sans_Brush_micelles_MCMC_%j.err   # Standard error log
#SBATCH --nodes=1                      # Number of nodes required
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for serial jobs)
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory required per node
#SBATCH --time=04:00:00                # Time limit (HH:MM:SS)
#SBATCH --partition=standard           # Cluster partition/queue to use

# 1. Load necessary environment modules
module purge
module load mamba


# 2. Activate your specific environment (if applicable)
./make_env.sh

# 3. Navigate to the working directory (optional, SLURM defaults to the submission directory)
cd $SLURM_SUBMIT_DIR

# 4. Execute the script
python3 cluster_job.py 