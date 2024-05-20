# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python [conda env:planet_orders]
#     language: python
#     name: conda-env-planet_orders-py
# ---

from pathlib import Path
import os
import rasterio as rio
import numpy as np
from xml.dom import minidom
import pandas as pd
from functools import reduce
import re

###Clearness quality checks:

##  Check cloud based on metadata:
def cloudest_meta(xmlfile):
    '''
    Get percent clouds from metadata (this is what Planet uses to filter orders)
    '''
    try:
        metaparse = minidom.parse(xmlfile)
    except:
        print("can't read xml file for {}".format(xmlfile))
    else:
        clouds_meta = metaparse.getElementsByTagName("opt:cloudCoverPercentage")[0].firstChild.data
    
    return clouds_meta

##  Check clouds based on 'clear'/'unclear' pixels in original udm (band 8 in udm2):
def cloudmask_udm(udmfile):
    '''
    Get cloud mask based on original unusable data mask file (original udm or band 8 of udm2)
    This will mask out unclear pixels as well as blackfilled data (data outside of actual imagery)
    '''
    if 'udm2' in udmfile:
        try:
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read(8)
            udm_mask = udm_array != 0
        except:
            print("could not read udm2 file for {}".format(udmfile))
            pass
    else:
        try:
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read()
            udm_mask = udm_array != 0
        except:
            print("could not read udm file for {}".format(udmfile))
            pass
            
    return udm_mask

def cloudest_udm(udmfile):
    '''
    Get percent clouds from original unusable data mask file (original udm or band 8 of udm2)
    In udm & udm2_band8, clear pixels == 0, blackfilled data == 1, and unclear pixels >1.
    This method removes blackfill from total before calculating percent unclear.
    '''
    if 'udm2' in udmfile:
        try:
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read(8)
        except:
            print("could not read udm2 file for {}".format(udmfile))
            return 999               
        else:
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read(8)
            num_pixels_udm = np.size(udm_array)
            num_black = np.count_nonzero(udm_array==1)
            num_validpix = num_pixels_udm - num_black
            num_notclear_udm = np.count_nonzero(udm_array > 1)
            percent_notclear_udm = 100 - (100 * (num_validpix - num_notclear_udm) / num_validpix)
            return percent_notclear_udm
        
    else:
        try:
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read()
        except:
            print("could not read udm file for {}".format(udmfile))
            return 999
        else:   
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read()
            num_pixels_udm = np.size(udm_array)
            num_black = np.count_nonzero(udm_array==1)
            num_validpix = num_pixels_udm - num_black
            num_notclear_udm = np.count_nonzero(udm_array > 1)
            percent_notclear_udm = 100 - (100 * (num_validpix - num_notclear_udm) / num_validpix)
            return percent_notclear_udm

##   Based on 'clear'/'unclear' in udm2_(band 1):
def cloudmask_udm2(udmfile):
    '''
    Get cloud mask from usable data mask file. Returns 3 different mask options.
    [0] is the uncear mask that is the sum of all mask bands in the udm2 file (also on band 1)
    [1] is the cloud/shadow mask only
    [2] is the cloud/shadow mask (bands 3 & 6) + the original udm mask (band 8)
    [3] is blackfill (outside of actual imagery. may need to add to [2] and [3])
    For band 1: clear =1, notclear = 0, blackfill = 0.
    For cloud and shadow bands: notclear = 1. clear =0, but backfill also =0.
    '''
    if 'udm2' in udmfile:
        try:
            with rio.open(udmfile, 'r') as src:
                udm2_array = src.read(1)
                orig_udm = src.read(8)
                cloud_band = src.read(6)
                shadow_band = src.read(3)
        except:
            print("could not read udm2 file for {}".format(udmfile))
            return 999, 999, 999
        
        else:
            mask_udm2notclear = udm2_array == 0
            orig_mask =  orig_udm != 0
            cloud_mask = cloud_band != 0
            shadow_mask = shadow_band != 0
            mask_udm2cs = cloud_mask | shadow_mask
            mask_udm2csorig = cloud_mask | shadow_mask | orig_mask
            blackfill = orig_udm == 1 
            return mask_udm2notclear, mask_udm2cs, mask_udm2csorig, blackfill
            
    else:
        print ("This is not a udm2 file, skipping udm2 calcs")
        pass
        

def cloudest_udm2(udmfile):
    '''
    Get percent unclear from usable data mask file. Returns 3 estimates:
    [0] is the uncear mask that is the sum of all mask bands in the udm2 file (also on band 1)
    [1] is the cloud/shadow mask only
    [2] is the cloud/shadow mask (bands 3 & 6) + the original udm mask (band 8)
    For band 1: clear =1, notclear = 0, blackfill = 0.
    For cloud and shadow bands: notclear = 1. clear =0, but backfill also =0.
    '''
    try:
        with rio.open(udmfile, 'r') as src:
            udm2_array = src.read(1)
            udm_array = src.read(8)
            cloud_band = src.read(6)
            shadow_band = src.read(3)
    except:
        print("could not read udm2 file for {}".format(udmfile))
        return 999, 999, 999
    
    else:
        num_pixels_udm = np.size(udm_array)
        num_black = np.count_nonzero(udm_array==1)
        num_validpix = num_pixels_udm - num_black
    
        num_clear_udm2notclear = np.count_nonzero(udm2_array)
        percent_masked_udm2notclear = 100 * (num_validpix - num_clear_udm2notclear) / num_validpix

        mask_udm2cs = cloud_band + shadow_band    
        num_masked_udm2cs = np.count_nonzero(mask_udm2cs)
        percent_masked_udm2cs = 100 - (100 * (num_validpix - num_masked_udm2cs) / num_validpix)
    
        orig_mask = udm_array > 1
        mask_udm2csorig = cloud_band + shadow_band + orig_mask
        num_masked_udm2csorig = np.count_nonzero(mask_udm2csorig)
        percent_masked_udm2csorig = 100 - (100 * (num_validpix - num_masked_udm2csorig) / num_validpix)
    
        return percent_masked_udm2notclear, percent_masked_udm2cs, percent_masked_udm2csorig

def print_cloud_comparisons(xmlfile, udmfile):
    print("according to metadata, cloud percentage is: {}%".format(cloudest_meta(xmlfile)))
    print("according to udm_band, {0:.01f}% of pixels are NOT clear.".format(cloudest_udm(udmfile)))
    print("according to udm2_band 1, {0:.01f}% of pixels are NOT clear.".format(cloudest_udm2(udmfile)[0]))
    print("according to udm2 cloud/shadow bands, {0:.01f}% of pixels are NOT clear.".format(cloudest_udm2(udmfile)[1]))
    print("according to udm2 cloud/shadow/udm bands, {0:.01f}% of pixels are NOT clear.".format(cloudest_udm2(udmfile)[2]))

def list_cloud_comparisons(in_dir, out_dir):
    '''
    Outputs a csv file with cloud estimates from various sources:
    'metadata' is reading from xml metadata (the same Planet uses to filter
    'percent_notclear_udm_1' is original udm estimate (everything unclear) from original udm file
    'percent_notclear_udm_2' is original udm estimate (everything unclear from band 8 of udm file
    'percent_notclear_udm2' is anything not clear (cloud, shados, light haze, heavy haze, etc.) from band 1 of udm2 file)
    'percent_cs_udm2' is cloud and shadow bands from udm2
    'percent_csorig_udm2' is cloud and shadow bands from udm2 plus original udm mask
    
    Planet files names are inconsistent; here is cuts it off at 21 digits, but may need to extract name with re package
    '''
    clouds_meta = []
    percent_notclear_udm_1 = []
    percent_notclear_udm_2 = []
    percent_notclear_udm2 = []
    percent_cs_udm2 = []
    percent_csorig_udm2 = []
    
    for f in os.listdir(in_dir):
        base_id = str(os.path.basename(f)[:21]) #Are the first 21 digits always unique???
        file = os.path.join(in_dir,f)
        if f.endswith('.xml'):
            cloud_est = {'img_id': base_id, 'clouds_meta':cloudest_meta(file)}
            clouds_meta.append(cloud_est)
        elif 'udm2' in f:
            cloud_est1 = {'img_id': base_id, 'percent_notclear_udm_2':cloudest_udm(file)}
            cloud_est2 = {'img_id': base_id, 'percent_notclear_udm2':cloudest_udm2(file)[0]}
            cloud_est3 = {'img_id': base_id, 'percent_cs_udm2':cloudest_udm2(file)[1]}
            cloud_est4 = {'img_id': base_id, 'percent_csorig_udm2':cloudest_udm2(file)[2]}
            percent_notclear_udm_2.append(cloud_est1)
            percent_notclear_udm2.append(cloud_est2)
            percent_cs_udm2.append(cloud_est3)
            percent_csorig_udm2.append(cloud_est4)
        elif 'udm' in f:
            cloud_est = {'img_id': base_id, 'percent_notclear_udm_1':cloudest_udm(file)}
            percent_notclear_udm_1.append(cloud_est)
     
    df1 = pd.DataFrame(clouds_meta)
    df2 = pd.DataFrame(percent_notclear_udm_1)
    df3 = pd.DataFrame(percent_notclear_udm_2)
    df4 = pd.DataFrame(percent_notclear_udm2)
    df5 = pd.DataFrame(percent_cs_udm2)
    df6 = pd.DataFrame(percent_csorig_udm2)
    all_est = [df1, df2, df3, df4, df5, df6]
    
    cloud_ests = reduce(lambda  left,right: pd.merge(left,right,on=['img_id'],
                                            how='outer'), all_est)

    outfile = os.path.join(out_dir, 'cloudEstimate_comparison.txt')
    pd.DataFrame.to_csv(cloud_ests, outfile, sep=',', na_rep='.', index=False)
# -


