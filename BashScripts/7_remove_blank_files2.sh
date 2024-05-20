#!/bin/bash  -l
#SBATCH --nodes=1 --ntasks-per-node 12
#  this is asking for 1 node, with 12 cores per node
#     the -l is needed on first line if you want to use modules
#SBATCH --job-name=rmblanksburntfields
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=rmBlanks.%j.out # STDOUT
#SBATCH --error=rmBlanks.%j.err # STDERR 

#Settables:
OUT_DIR="<path/to/image_dir/planet>/SR_poly_2"
ENDSTRING=".tif"

#Activate the virtual environment (which relys on anaconda)
conda activate conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR

burntfields rm_blanks --out_dir $OUT_DIR --endstring $ENDSTRING
