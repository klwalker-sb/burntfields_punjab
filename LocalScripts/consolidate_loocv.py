#!/usr/bin/env python
# coding: utf-8

'''
Use this script to recombine holdout and prediction datasets from LOOCV on cluster before downloading
TODO: parallelize original LOOCV script correctly and integrate this into single script.
'''

import os
import csv
import pandas as pd
import numpy as np

in_dir = "path/to/out_dir/LOOCV"
out_dir = "path/to/out/dir"
data_dir = "path/to/data_dir"
labeled_list = os.path.join(data_dir,'labels_2019_with_bbox_V4.csv')
all_features = os.path.join(data_dir,'V4_pixelData_COMBO.csv')

ho = []
full_pred_df = pd.read_csv(all_features, usecols=["pixel_id"])
subfolders = [f.name for f in os.scandir(in_dir) if f.is_dir() ]
for sf in subfolders:
    pred_df = pd.read_csv(os.path.join(in_dir,sf,'LOOCV_All_Pred.csv'),usecols=["meanPred", "pixel_id"])
    full_pred_df = full_pred_df.merge(pred_df,how='left',left_on='pixel_id', right_on='pixel_id')
    full_pred_df.rename(columns={'meanPred': 'meanPred_'+str(sf)}, inplace=True)
    ho_df = pd.read_csv(os.path.join(in_dir,sf,'LOOCV_HO_Pred.csv'),usecols=["meanPred", "pixel_id"])
    ho.append(ho_df)

ho_out = pd.concat(ho)

pd.DataFrame.to_csv(full_pred_df, os.path.join(out_dir,'LOOCV_PixelPredictions_Full.csv'), sep=',', na_rep='NaN', index=False)
pd.DataFrame.to_csv(ho_out, os.path.join(out_dir,'LOOCV_Holdout_Predictions.csv'), sep=',', na_rep='NaN', index=False)

