#!/bin/bash
# SBATCH -N 1 #number of nodes
# SBATCH -n 2 #number of cores per node
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=<your_email>
# SBATCH --error=LOOCV.%A.%a.%j.err
# SBATCH --output=LOOCV.%A.%a.%j.out 
# SBATCH --array=19
# SBATCH --mem-per-cpu=32G   # memory per cpu-core
# sleep $((RANDOM%30+1))

#Settables
OUT_DIR="<path/to/out_dir>/Punjab_LOOCV/Results/${SLURM_ARRAY_TASK_ID}"
HO_LIST="<path/to/out_dir>/Punjab_LOOCV/HOSets/${SLURM_ARRAY_TASK_ID}.csv"
LABELS="<path/to/data_dir>/labels_2019_with_bbox_V4.csv"
ALLFEAT="<path/to/data_dir>/V4_pixelData_COMBO.csv"
VARIABLES="<path/to/data_dir>/Variables.csv"
SEED1=8
SEED2=6888
DROP_BORDER='True'

conda activate venv.burntfields
cd $SLURM_SUBMIT_DIR
burntfields loocv --out_dir $OUT_DIR --ho_list $HO_LIST --labeled_list $LABELS --variable_path $VARIABLES --all_features $ALLFEAT --seed1 $SEED1 --seed2 $SEED2 --drop_border $DROP_BORDER
