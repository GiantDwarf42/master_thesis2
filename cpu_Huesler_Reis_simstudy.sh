#!/bin/sh
#SBATCH --job-name Huesler_Reis            # this is a parameter to help you sort your job when listing it
#SBATCH --error protocols/testrun-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output protocols/testrun-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 10             # number of cpus for each task. One by default
#SBATCH --array=1-200

#SBATCH --partition shared-cpu 	# the partition to use. By default debug-cpu		
    				
				
#SBATCH --mem=1G                      #memory needed (default 3GB)
#SBATCH --time 300:00                  # maximum run time.


module purge                           #Unload all modules


module load Anaconda3              # load Anaconda





## define and create a unique scratch directory
#SCRATCH_DIRECTORY=/global/work/${USER}/kelp/${SLURM_JOBID}
#mkdir -p ${SCRATCH_DIRECTORY}
#cd ${SCRATCH_DIRECTORY}

echo "before calling source: $PATH"
echo "Home: $HOME"



conda env list
# Activate Anaconda work environment
# source $HOME/.bashrc
source activate mt1

#conda deactivate


echo "after calling source: $PATH"

conda env list

# Print the job ID and array index
echo "Job ID: $SLURM_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

# Your calculation or command goes here
echo "Performing calculation for task $SLURM_ARRAY_TASK_ID"


srun python Huesler_Reis_simstudy_server.py $SLURM_ARRAY_TASK_ID              
# run your software

pwd

echo "executed"