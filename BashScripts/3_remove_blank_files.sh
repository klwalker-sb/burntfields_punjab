#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node 12
#SBATCH --job-name=rmblanksburntfields
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=rmBlanks.%j.out # STDOUT
#SBATCH --error=rmBlanks.%j.err # STDERR 

#Settables:
OUT_DIR="<path/to/image_dir/planet>/SR_clipped"
ENDSTRING=".tif"

#Activate the virtual environment (which relys on anaconda)
conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR

burntfields rm_blanks --out_dir $OUT_DIR --endstring $ENDSTRING
