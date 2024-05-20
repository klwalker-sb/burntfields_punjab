#!/bin/bash 
#SBATCH --nodes=1 --ntasks-per-node 12
#  this is asking for 1 node, with 12 cores per node
#     the -l is needed on first line if you want to use modules
#SBATCH --job-name=MaskClouds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=MaskClouds.%j.out # STDOUT
#SBATCH --error=MaskClouds.%j.err # STDERR 

#Settables:
OUT_DIR="<path/to/image_dir/planet>/SR_ClipCalMasked"
IN_DIR="<path/to/image_dir/planet>/SR_clipped"
TYPE='udm2csorig'
ENDSTRING='AnalyticMS_SR'

#Activate the virtual environment (which relys on anaconda)
conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR

burntfields mask --in_dir $IN_DIR --out_dir $OUT_DIR --out_type $TYPE --endstring $ENDSTRING
