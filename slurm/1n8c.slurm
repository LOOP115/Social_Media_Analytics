#!/bin/bash
#SBATCH --job-name="Social_Media_Analysis"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-1:00:00
#SBATCH --output=1n8c.out

# Specify your email address to be notified of progress.
# Includes job usage statistics
#SBATCH --mail-user=jiahchen4@student.unimelb.edu.au
#SBATCH --mail-type=ALL


# Load required modules
module purge

module load python/3.7.4
module load mpi4py/3.0.2-timed-pingpong

mkdir ~/virtualenv
virtualenv ~/virtualenv/python3.7.4
source ~/virtualenv/python3.7.4/bin/activate
pip install numpy
pip install pandas


# Launch multiple process python code
echo "Processing tinyTwitter"
time mpirun -np 8 python3 main.py tiny
echo "Processing smallTwitter"
time mpirun -np 8 python3 main.py small
echo "Processing bigTwitter"
time mpirun -np 8 python3 main.py big

deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
