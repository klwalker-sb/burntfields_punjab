#!/bin/bash 
#SBATCH -N 1 #number of nodes
#SBATCH -n 1 #number of cores per node
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --error=clipToPoly.%A.%a.%j.err
#SBATCH --output=clipToPoly.%A.%a.%j.out 
#SBATCH --array=1-30
#sleep $((RANDOM%30+1))

#Settables
OUT_DIR="<path/to/image_dir/planet>/TS_polys"
IN_DIR="<path/to/image_dir/planet>/SR_ClipCalMasked"
SHAPEFILE="<path/to/data_dir>/clean_2Tp_example.shp"
FILTER="<path/to/data_dir>/bbox_AOIs/${SLURM_ARRAY_TASK_ID}.csv"
NUM_CORES=1
STRING='AnalyticMS_SR'

conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR

burntfields clip_to_poly --shapefile $SHAPEFILE --shape_filter $FILTER --in_dir $IN_DIR --out_dir $OUT_DIR  --mstring $STRING --num_cores $NUM_CORES

