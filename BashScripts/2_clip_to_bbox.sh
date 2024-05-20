#!/bin/bash
#SBATCH -N 1 #number of nodes
#SBATCH -n 4 #number of cores per node
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --error=clip.%A.%a.%j.err
#SBATCH --output=clip.%A.%a.%j.out 
#SBATCH --array=1-14
#sleep $((RANDOM%30+1))

#Settables
OUT_DIR="<path/to/image_dir/planet>/SR_clipped"
IN_DIR="<path/to/image_dir/planet>/SR_Harmonized"
#AOI_FILE="/path/to/data_dir/lon_lat_bbox_600m_example.csv"
AOI_sub="/path/to/data_dir/bbox_AOIs/${SLURM_ARRAY_TASK_ID}.csv"

conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR

burntfields clip_images --aoi_file $AOI_sub --in_dir $IN_DIR --out_dir $OUT_DIR

