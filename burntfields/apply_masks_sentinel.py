#!/usr/bin/env python
# coding: utf-8 

from pathlib import Path
import os
import rasterio as rio
import numpy as np
import pandas as pd

##For local running/testing only
#in_dir = "E:/Punjab/PunjabSentinel/2019/L2A_20m"
#out_dir = "E:/Punjab/PunjabSentinel/2019/L2A_20m/Masked_v2"

def cloudmask_scl(maskfile, out_type):
    '''
    Get cloud mask from SCL file
    0= NoData, 1= Saturated/Defective, 2-Dark Areas, 3= Cloud Shadows, 4=Vegetation, 5=Bare Soil,
    6= Water, 7= Clouds, low probability, 8= Clouds, medium probability, 9= Clouds, high probability, 
    10= Cirrus clouds, 11=snow
    TODO: Fix method to write mask file as binary raster to enable future touchup of cloudmask
    TODO: fix GDAL to open jp2 files
    WORKAROUND: jp2 files aren't currently being read in my environment, so I opened them in ArcGIS and created
       the binary mask there. I named these files '..._SCLlight_binary', so the 'binary' clause works on them.
    '''
    if 'binary' in maskfile:
        try:
            with rio.open(maskfile, 'r') as src:
                scl_array = src.read(1)
            return scl_array != 0       
        except:
            print ('cannot read file {}'.format(maskfile))
            pass

    if 'GEE' in maskfile:
        try:
            with rio.open(maskfile, 'r') as src:
                scl_array = src.read(1)
            return scl_array != 0       
        except:
            print ('cannot read file {}'.format(maskfile))
            pass

    elif 'SCL' in maskfile:
        try:
            with rio.open(maskfile, 'r') as src:
                scl_array = src.read(1)
            if out_type == 'SCLlight':
                cloud_mask = (scl_array == 8 | scl_array == 9)
            elif out_type == 'SCLFull':
                cloud_mask = (scl_array == 2 | scl_array == 8 | scl_array == 9 | scl_array == 10)
            else:
                print: ("invalid out type")

            shadow_mask = (scl_array == 2) #3-dark areas often select burned areas, so do not mask
            lowqual_mask = (scl_array == 0 | scl_array == 1 | scl_array == 11)
            mask_scl = cloud_mask | shadow_mask | lowqual_mask

            #meta = src.meta
            #with rio.open(masked_file, 'w', **meta) as dst: 
                #dst.write(src, masked_band)

            return mask_scl

        except:
            print ('cannot read file {}'.format(maskfile))
            pass

    else:
        print ("This is not a scl file, skipping scl calcs")
        pass

def apply_mask_to_sentinel(ras_stack, maskfile, out_dir, out_type):
    '''
    creates a new image with the cloud mask applied based on scl.
    This function actually writes a new file to the output directory... (consider a more efficient alternative)
    Masking options are:
    '''
    img_id = str(os.path.basename(ras_stack).strip('.tif'))
    if out_type == 'SCLlight' or out_type == 'SCLfull' or out_type == 'binary' or out_type == "GEECloudProb":
        masky = (cloudmask_scl(maskfile, out_type))
        if out_type == 'binary':
            masked_file = os.path.join(out_dir,img_id+'_masked_SCLlight.tif')
        elif out_type == 'GEECloudProb':
            masked_file = os.path.join(out_dir,img_id+'_GEECloudProb.tif')
    else:
        print ("{} is not currently an out_type option, Choose 'SCLlight, SCLfull, binary or GEECloudProb".format(out_type))

    if os.path.isfile(masked_file):
        print ("{} already exists".format(masked_file))
    else:
        try:
            with rio.open(ras_stack, 'r') as src:
                samp_band = src.read(1)
        except: 
            print ("could not open raster {}".format(ras_stack))
            return 999
            pass
        else:    
            ### Get percent masked pixels for table: TODO: fix these calculations!
            valid_pix = np.count_nonzero(samp_band)
            print('there are {} valid pixels in this raster'.format(valid_pix))
            if valid_pix == 0:
                per_masked = 100
            else:
                masked_band = np.ma.masked_array(samp_band, masky)
                masked_band = masked_band.filled(src.nodata)
                remain_pix = np.count_nonzero(masked_band)
                #print('there are {} pixels remaining after masking'.format(remain_pix))
                per_masked = 100 * (valid_pix - remain_pix)/valid_pix
                #print('per_masked = {}'.format(per_masked))

            ### Create masked band
                masked_stack = []
                for b in range(1,10):
                    with rio.open(ras_stack, 'r') as src:
                        bandx = src.read(b)
                    ### Apply mask in binary form (0=clear) to output (add mask to no data of image files (0=nodata)
                    masked_band = np.ma.masked_array(bandx, mask=masky)
                    ## need to promote to float 32 to apply mask with NaNs
                    masked_band = masked_band.astype(rio.float32)
                    ## If keeping NaNs, use np.nan instead of src.nodata. Here src.nodata (0=nodata) is okay
                    masked_band = masked_band.filled(np.nan)
                    ## Demoting again to integer... I think there is a more efficient way to do this.
                    #masked_band = masked_band.astype(rio.uint16)
                    # Write band calculations to new raster files
                    meta = src.meta
                    masked_stack.append(masked_band)
                    ## Need to update meta from original uint16 to float32 if keeping NaNs
                    meta.update(dtype=rio.float32, nodata=0)

                with rio.open(masked_file, 'w', **meta) as dst: 
                    for band_nr, src in enumerate(masked_stack, start=1):
                        dst.write(src, band_nr)

            return per_masked

def mask_images_sentinel(in_dir, out_dir, string, out_type):
    '''
    string is the string of the images files to be masked and output (i.e. '')
    OutType is type of mask to apply. Options are:
        SCLlight: Does not mask low cloud probability or cirrus
    Also creates cvs file with image name and %masked for possible post-hoc filtering
    TODO: Fix calculations for % masked
    '''

    mask_options = ['SCLlight', 'SCLfull', 'binary', 'GEECloudProb']
    if out_type not in mask_options:
        print("Currently supported mask types are: 'SCLlight', 'SCLfull', 'binary', 'GEECloudProb'. {} does not match.".format(out_type))
    per_masked_list = []

    for img in os.listdir(in_dir):
        if img.endswith(string):
            base_id = str(os.path.basename(img)[:22])
            img_path = os.path.join(in_dir,img)
            if out_type == 'binary':
                mask_path = os.path.join(in_dir,base_id+'_Mask_SCLlight_binary.tif')
            elif out_type == 'GEECloudProb':
                mask_path = os.path.join(in_dir,base_id+'_Mask_GEEcloudProb50.tif')
            per_masked = apply_mask_to_sentinel(img_path, mask_path, out_dir, out_type)
            per_masked_entry = {'img_id': base_id, 'percentMasked':per_masked}
            per_masked_list.append(per_masked_entry)

    df1 = pd.DataFrame(per_masked_list)
    outfile = os.path.join(out_dir, 'percentMasked_out.txt')
    pd.DataFrame.to_csv(df1, outfile, sep=',', na_rep='.', index=False)

###To run locally:
#mask_images_Sentinel (in_dir, out_dir, 'B2345678A1112.tif', 'binary')
#mask_images_Sentinel (in_dir, out_dir, 'B2345678A1112.tif', 'GEECloudProb')
