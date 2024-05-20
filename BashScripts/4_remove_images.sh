#!/bin/bash 
#SBATCH --nodes=1 --ntasks-per-node 12
#  this is asking for 1 node, with 12 cores per node
#SBATCH --job-name=filter_images
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=filter_Images.%j.out # STDOUT
#SBATCH --error=filter_images.%j.err # STDERR 

#Settables:
IN_DIR="<path/to/image_dir/planet>/SR_clipped"
OUT_DIR="<path/to/image_dir/planet>/SR_clipped_2"
REMOVE_FILE="<path/to/data_dir>/cloudy_scenes.txt"

#Activate the virtual environment (which relys on anaconda)
conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR

burntfields rm_images --rm_file $REMOVE_FILE --in_dir $IN_DIR --out_dir $OUT_DIR
