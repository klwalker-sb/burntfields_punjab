#!/usr/bin/env python
# coding: utf-8
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio as rio
import rasterio.warp
import json
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]
tif_list = sys.argv[3]

def make_extent_xml(in_dir, out_dir, tif_list):
    '''
    Makes xml with entries to match those for Planet bounding coordinates
    so that geotiff files missing xml file can be processed in same way as others.  
    '''    
    tiffs_todo = pd.read_csv(tif_list, index_col=None)
    for tif in tiffs_todo['missing']:
        for f in os.listdir(in_dir):
            if tif in f and f.endswith('AnalyticMS_SR_harmonized.tif'):
                with rio.open(os.path.join(in_dir,f)) as src:
                    bounds = src.bounds
                    fbounds = rasterio.warp.transform_bounds(src.crs,{'init': 'epsg:4326'}, *bounds)
                    #bound_dict = bounds._asdict()
                    bound_dict={'left':fbounds[0],'bottom':fbounds[1],'right':fbounds[2],'top':fbounds[3]}
                    print(bound_dict)
                    with open(os.path.join(out_dir,tif+'fake_xml.json'),'w') as fp:
                        json.dump(bound_dict, fp, sort_keys=True, indent=4)


make_extent_xml(in_dir, out_dir, tif_list)



