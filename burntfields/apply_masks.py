#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os
import rasterio as rio
import numpy as np
from xml.dom import minidom
import pandas as pd
from functools import reduce
import re

###Get Cloud Masks

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
            return udm_mask
        except:
            print ('WARNING: cannot read file {}; cloudmask not applied'.format(udmfile))
            pass
    else:
        try:
            with rio.open(udmfile, 'r') as src:
                udm_array = src.read()
            udm_mask = udm_array != 0
            return udm_mask
        except:
            print ('WARNING: cannot read file {}; cloudmask not applied'.format(udmfile))
            pass

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

            mask_udm2notclear = udm2_array == 0
            orig_mask =  orig_udm != 0
            cloud_mask = cloud_band != 0
            shadow_mask = shadow_band != 0
            mask_udm2cs = cloud_mask | shadow_mask
            mask_udm2csorig = cloud_mask | shadow_mask | orig_mask
            blackfill = orig_udm == 1

            return mask_udm2notclear, mask_udm2cs, mask_udm2csorig, blackfill

        except:
            print ('cannot read file {}'.format(udmfile))
            pass

    else:
        print ("This is not a udm2 file, skipping udm2 calcs")
        pass


def apply_mask_to_planet_sr(sr_image, udmfilename, out_dir, out_type):
    '''
    creates a new 4-band SR with the UDM2 mask applied.
    This function actually writes a new file to the output directory... (consider a more efficient alternative)
    Masking options are:
       udm = mask based on original udm file (with all quality issues masked)
       udm2all = mask based on 1st band of udm2 file (with all files masked)
       udm2cs = mask based on cloud/shadow bands of udm2
       udm2csorig = mask based on cloud/shadow bands of udm2 + original udm mask
    '''
    img_id = str(os.path.basename(sr_image).strip('.tif'))
    empty_ras = []
    if out_type == 'udm2all':
        try:
            masky = cloudmask_udm2(udmfilename)[0]
        except:
            pass
        else:
            masked_sr_file = os.path.join(out_dir,img_id+'_masked_udm2all.tif')
    elif out_type == 'udm2cs':
        try:
            masky = cloudmask_udm2(udmfilename)[1]
        except:
            pass
        else:
            masked_sr_file = os.path.join(out_dir,img_id+'_masked_udm2cs.tif')
    elif out_type == 'udm2csorig':
        try:
            masky = cloudmask_udm2(udmfilename)[2]
        except:
            pass
        else:
            masked_sr_file = os.path.join(out_dir,img_id+'_masked_udm2csorig.tif')
    elif out_type == 'udm':
        if "udm2" in udmfilename:
            try:
                masky = cloudmask_udm(udmfilename)
            except:
                pass
            else:
                masked_sr_file = os.path.join(out_dir,img_id+'_masked.tif')
        elif "udm" in udmfilename:
            try:
                masky = cloudmask_udm(udmfilename)
            except:
                pass
            else:
                masked_sr_file = os.path.join(out_dir,img_id+'_masked.tif')
        else:
            print ("mask filename must contain udm or udm2")
    else:
        print ("{} is not currently an out_type option, Choose 'udm', 'udm2all', 'utm2cs', or 'utm2orig'".format(out_type))

    try:
        if os.path.isfile(masked_sr_file):
            print ("{} already exists".format(masked_sr_file))
            pass
    except:
            pass
    else:
        # Need to mask and output all 4 bands, but first get perMask value from first band:
        try:
            with rio.open(sr_image, 'r') as src:
                samp_band = src.read(1)
        except: 
            print ("could not open raster {}".format(sr_image))
            return 999
            pass
        else:    
            ### Get percent masked pixels for table:
            valid_pix = np.count_nonzero(samp_band)
            if valid_pix == 0:
                per_masked = 100
                pass
            else:
                masked_band = np.ma.masked_array(samp_band, mask=masky)
                masked_band = masked_band.filled(src.nodata)
                remain_pix = np.count_nonzero(masked_band)
                per_masked = 100 * (valid_pix - remain_pix)/valid_pix

                ### Create masked band
                masked_stack = []
                for b in range(1,5):
                    with rio.open(sr_image, 'r') as src:
                        bandx = src.read(b)
                    ### Apply mask in binary form (0=clear) to output (add mask to no data of image files (0=nodata)
                    masked_band = np.ma.masked_array(bandx, mask=masky)
                    ## need to promote to float 32 to apply mask with NaNs
                    masked_band = masked_band.astype(rio.float32)
                    ## If keeping NaNs, use np.nan instead of src.nodata. Here src.nodata (0=nodata) is okay
                    masked_band = masked_band.filled(src.nodata)
                    ## Demoting agoin to integer... I think there is a more efficient way to do this.
                    masked_band = masked_band.astype(rio.uint16)
                    masked_stack.append(masked_band)
                    # Write band calculations to new raster files
                    meta = src.meta
                    ## Need to update meta from original uint16 to float32 if keeping NaNs
                    #meta.update(dtype=rio.float32, nodata=0)

                with rio.open(masked_sr_file, 'w', **meta) as dst: 
                    for band_nr, src in enumerate(masked_stack, start=1):
                        dst.write(src, band_nr) 

            return per_masked


def mask_images(in_dir, out_dir, endstring, out_type):
    '''
    creates a new 4-band image with the UDM2 mask applied.
    (images and corresponding udm/udm2 masks should be in same input dirrectory)
    Endstring is the ending of the images files to be masked and output (i.e. 'SR_clip.tif')
    out_type is type of mask to apply. Options are:
       udm = mask based on original udm file (with all quality issues masked)
       udm2all = mask based on 1st band of udm2 file (with all files masked)
       udm2cs = mask based on cloud/shadow bands of udm2
       udm2csorig = mask based on cloud/shadow bands of udm2 + original udm mask
    Also creates cvs file with image name and %masked for possible post-hoc filtering
    '''

    mask_options = ['udm','udm2all','udm2cs','udm2csorig']
    if out_type not in mask_options:
        print("Currently supported mask types are: 'udm' 'udm2all' 'udm2cs', 'udm2csorig'. {} does not match.".format(out_type))
    per_masked_list = []

    files = [os.path.join(in_dir,f) for f in os.listdir(in_dir) if f.endswith(endstring)]
    for img in files:
        
        ##Note, in original images, [:21] is used:
        #base_id = str(os.path.basename(img)[:21])
        ##but for clipped images, polygon id is added to front, so need 21 + 11:
        base_id = str(os.path.basename(img)[:32])
        img_path = os.path.join(in_dir,img)
        if out_type == 'udm':
            if 'clip' in os.path.basename(img):     
                mask_path = img_path.split(endstring, 1)[0]+'udm_clip.tif'
            else:
                mask_path = img_path.split(endstring, 1)[0]+'udm.tif'
        else:
            if 'clip' in os.path.basename(img):
                mask_path = img_path.split(endstring, 1)[0]+'udm2_clip.tif'
            else:
                mask_path = img_path.split(endstring, 1)[0]+'udm2.tif'

        per_masked = apply_mask_to_planet_sr(img_path, mask_path, out_dir, out_type)
        per_masked_entry = {'img_id': base_id, 'percentMasked':per_masked}
        per_masked_list.append(per_masked_entry)

    df1 = pd.DataFrame(per_masked_list)
    outfile = os.path.join(out_dir, 'percentMasked_out.txt')
    pd.DataFrame.to_csv(df1, outfile, sep=',', na_rep='.', index=False)

##endstring = 'SR_clip', 'AnalyticMS_SR, 'AnalyticMS_SR_harmonized', etc...
#mask_images (in_dir, out_dir, 'SR_clip', 'udm2csorig')
