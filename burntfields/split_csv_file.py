#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd


#For local testing:
#aoi_file = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_1_input_transformed/labels_2019_with_bbox_V5.csv"
aoi_file = "C:/Users/klobw/Desktop/Testing/labels_V5.csv"
out_aoi_dir = "C:/Users/klobw/Desktop/Testing/HOSets5"


def split_csv(aoi_file, out_dir, nrows):
    
    fullfile = pd.read_csv(aoi_file)
    num_files = int(len(fullfile)/nrows +1)
 
    for i in range (num_files):
        df = fullfile[nrows*i:nrows*(i+1)]
        df.to_csv(os.path.join(out_dir,'{}.csv'.format(i+1)), index=False)


#split_csv(aoi_file, out_aoi_dir, 20)

