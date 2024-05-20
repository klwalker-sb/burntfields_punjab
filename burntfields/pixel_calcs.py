#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import osgeo
import rasterio as rio
import sklearn
import csv
import operator
from affine import Affine
from pyproj import Proj, transform


def get_avg_poly_stats(ids, data_dir):
    '''
    Not currently using. Finish if needed
    (because we did pixel-level first, current field-level model just uses the pixel-level dataframes and aggregates)
    '''
    fields = pd.read_csv(ids)
    num_image_list = []

    ##make blank lists to store ids that are missing images
    blank_ids = []
    ##Set up lists to store final data:
    band_names = ['blue', 'green', 'red', 'nir']
    band_data = [[] for b in range(4)]

    for i in range(len(fields)):
        id = fields['unique_id'][i] ## getting the polygon id
        val_list = []
        print("Working on {} with id {}". format(i, id))
        img_count = 0
        ##For each image, Get Avg,Max,Min,Count of all pixels for each band + two indices
        for j in os.listdir(data_dir): 
            if str(id) in j:
                img_count = img_count+1
                input_filename = os.path.join(data_dir,j)
                try:
                    with rio.open(input_filename) as src:
                        B_red = src.read(3)
                except:
                    print("problem processing image {}. skipping.".format(j))
                    pass
                else:
                    ras = rio.open(os.path.join(input_filename), 'r')
                    #Open each band and get stats (1=blue, 2=green, 3=red, 4=NIR, as specified above)
                    b = 0
                    while b < len(band_names):
                        ras.np = ras.read(b+1)
                        maskedras = ma.masked_array(ras.np, mask=mask)
                        maskedras = maskedras.filled(ras.nodata)
                        affine = ras.transform
                        #Get summary statistics for all pixels
                        stats = zonal_stats(maskedras, maskedras, affine = affine, stats = [stat], geojson_out = True, nodata=0)
                        #Add summary statistics to list containing all images for each field (for each band)
                        val_list.append(stats[x])
                        x = x + 1
                        band_data[b].append(['properties'][stat])
                        b = b + 1
                    ##Summarize statistics from all images (Avg,Max,Min,Count,stdv) for field ID, for each band
                    
                    ##Add this to final dataframe containing field_ID and model variables
                    
                    ##Do for indices as well


def max_drop(arr, thresh, buf, strict_window=False):
    '''
    Get Max drop in time series vals, with filter options to control for clouds. 
         Thresh input determines how low the signal needs to remain to be counted as valid drop (i.e. <avg value?)
         and buf sets # of images it is sustained for
    If strictWindow=True, need to see two images consecutively (no NANs between)

    an alternative method would be to reverse the array (array = np.max(array) - array)
    and then use scipy_signals: find_peaks(array, distance=(3, np.max(array)))
    '''
    #Get window to look in for max; last images are needed for buffer if applied
    
    if strict_window == True:
        ##Only consider observations if they are actually adjacent sequentially (no NANs)
        seq = arr
        seqwin = arr[:len(arr)-buf] 
    else:
        ##Use sequence withous NaNs to provide more flexibility in dealing with NAs. 
        ##TODO: improve this to filter out number of days / images that can occur between observations
        seq = arr[~np.isnan(arr)]
        seq = seq.values
        seqwin = seq[:len(seq)-(buf+1)]
        
    diff = np.diff(seqwin)
    max_cng = np.nanmin(diff)
    cng_period = 0
    #print("original maxDrop = {}".format(MaxCng))
    confirmed = 0
    this_try = 0
    i = 0
    hit = False
    while confirmed == 0 and this_try < min(len(seqwin),10):
        for i in range(len(seqwin)-1):
            if seq[i+1]-seq[i]==max_cng:
                if buf == 0:
                    hit = True
                elif buf == 1:
                    ##note nan clauses now redundant if strictWindow=False (and if want strict window, don't want these...)
                    if ((seq[i+2] < thresh or (np.isnan(seq[i+2]) and seq[i+3] < thresh))):
                        hit = True
                elif buf == 2:
                    ##note nan clauses now redundant if strictWindow=False (and if want strict window, don't want these...)
                    if ((seq[i+2] < thresh or (np.isnan(seq[i+2]) and (seq[i+3] < thresh))) and ((seq[i+3] < thresh
                        or (np.isnan(seq[i+3])and seq[1+4] < thresh)))):
                        hit = True
                else: 
                    hit = False
                if hit == True: 
                    max_cng = max_cng
                    cng_period = i
                    confirmed = 1
                    i = i + 1
                else:               
                    diff2 = np.delete(diff, np.where(diff == max_cng), axis=0)
                    try:
                        max_cng = np.nanmin(diff2)
                    except:
                        this_try = 10
                        max_cng = max_cng
                    else:
                        diff = diff2
                        this_try = this_try+1
                        
    #print("buffered max_drop = {}, occurs at {}".format(max_cng,cng_period))              
    return(max_cng)

def max_spike(arr, thresh, buf, strict_window = False):
    '''
    Get Max spike in time series vals, with filter options to control for clouds. 
         Thresh input determines how low the signal needs to remain to be counted as valid spike (i.e. >avg value?)
         and buf sets # of images it is sustained for
    If strictWindow=True, need to see two images consecutively (no NANs between)

    an alternative method would be to use scipy_signals: find_peaks(array, distance=(3, np.max(array)))
    '''
    #Get window to look in for max (seqwin); last images are needed for buffer if applied
    
    if strict_window == True:
        ##Only consider observations if they are actually adjacent sequentially (no NANs)
        seq = arr
        seqwin = arr[:len(arr)-buf] 
    else:
        ##Use sequence withous NaNs to provide more flexibility in dealing with NAs. 
        ##TODO: improve this to filter out number of days / images that can occur between observations
        seq = arr[~np.isnan(arr)]
        seq = seq.values
        seqwin = seq[:len(seq)-buf]
        
    diff = np.diff(seqwin)
    max_cng = np.nanmax(diff)
    cng_period = 0
    #print("original max spike = {}".format(max_cng))
    confirmed = 0
    this_try = 0
    i = 0
    hit = False
    while confirmed == 0 and this_try < min(len(seqwin),10):
        for i in range(len(seqwin)-1):
            if seq[i+1]-seq[i]==max_cng:
                if buf == 0:  #TODO: This is holding place for future filter by observation window period
                    hit = True
                elif buf == 1:
                    ##note nan clauses now redundant if strictWindow=False (and if want strict window, don't want these...)
                    if ((seq[i+2] > thresh or (np.isnan(seq[i+2]) and seq[i+3] > thresh))):
                        hit = True
                elif buf == 2:
                    ##note nan clauses now redundant if strictWindow=False (and if want strict window, don't want these...)
                    if ((seq[i+2] > thresh or (np.isnan(seq[i+2]) and (seq[i+3] > thresh))) and ((seq[i+3] > thresh
                        or (np.isnan(seq[i+3])and seq[1+4] > thresh)))):
                        hit = True
                else: 
                    hit = False
                if hit == True:
                    max_cng = max_cng
                    cng_period = i
                    confirmed = 1
                    i = i+1
                else: 
                    #Otherwise discard this observation as noise and get next max
                    Diff2 = np.delete(diff, np.where(diff == max_cng), axis=0)
                    try:
                        max_cng = np.nanmax(Diff2)
                    except: #Can't do it if there is nothing left
                        this_try = 10
                        max_cng = max_cng #leave as old MaxCng if new doesn't work
                    else:
                        diff = Diff2
                        this_try = this_try + 1
                    
    #print("buffered maxSpike = {}, occurs at {}".format(max_cng,cng_period))              
    return(max_cng)


def get_pix_coords(ras_file,out_dir):
    '''
    Gets coordinates for all gridcells in rasterfile, then eliminates those masked in polygon file to 
    return only coordinates within polygon (with indices that match other iterations over raster below)
    Note it would probably be easier to process all polygons at once from one big shapefile, but the imagery is
    already all cropped to polygons.

    Returns dictionary with format: {id:{'Xcoord':x, 'Ycoord':y}}
    '''    
    with rio.open(ras_file) as ras:
        ras.blue = ras.read(1, masked=True)
    ##Get pixel ids & coordinates:
    T0 = ras.transform #this gives coord of top left corner of raster
    T1 = T0 * Affine.translation(0.5, 0.5) #this shifts coord from top corner to center of cell
    p1 = Proj(ras.crs) #to check coordinate system
    #print(p1)
    rc2xy = lambda ras, c: (c, ras) * T1 #gets coordinate for center of cell in x,y position
    #print(rc2xy(0, 0))
    w = ras.width
    h = ras.height
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    eastings, northings = np.vectorize(rc2xy, otypes=[np.float, np.float])(rows, cols)
    eastings = eastings.flatten()
    northings = northings.flatten() 
    bluevals = ma.masked_array(ras.blue, dtype='float64')
    bluevals = bluevals.flatten()
    polypix = {}
    for i in range(len(bluevals)):
        if bluevals[i] is ma.masked:
            pass
        else:
            polypix[i] = {}
            polypix[i]['Xcoord'] = eastings[i]
            polypix[i]['Ycoord'] = northings[i]
            
    #Nested dictionary to dataframe:
    coord_df = pd.DataFrame.from_dict(polypix,orient='index')
    pd.DataFrame.to_csv(coord_df, os.path.join(out_dir,'CoordCheck.csv'), sep=',', na_rep='NaN', index=True)
    
    return polypix


def poly_data_to_dict (ids, data_dir, out_dir, num_bands):
    '''
    For each polygon, get coordinates and list of images with all values for each band
    Output is nested distionary in the format:
    {PolyID:{Coords:{CoordX, CoordY}}Images:{0:{blue:{values},green:{values},red:{values},nir{values}},1:{..}....}}
    '''
    ## For each polygon in training list:
    fields = pd.read_csv(ids)
    num_image_list = []
    if num_bands == 3:
        band_names = ['blue', 'green', 'red']
    if num_bands == 4:
        band_names = ['blue', 'green', 'red', 'nir']
    elif num_bands == 9:
        band_names = ['blue', 'green', 'red', 'redEdge1', 'redEdge2', 'redEdge3', 'nir', 'SWIR1', 'SWIR2']
    poly_data = {} #Outer dictionary To contain polyID, image ids and arrays covering all pixels for 4 band TS
    ##make blank lists to store ids that are missing images
    blank_ids = []

    for i in range(len(fields)):
        id = fields['unique_id'][i] ## getting the polygon id
        print("Working on {} with id {}". format(i, id))
        poly_data[id] = {}
        img_count = 0
        goodimg_count = 0
        ##For each image in dataset:
        poly_data[id]['images']={}
        numpix=[]
        for j in os.listdir(data_dir):
            img_fid = j[:j.index("_")] #Gets first portion of name prior to first dash
            if str(id) == str(img_fid):
                ##Filter out images from last 2 weeks of year, in case downloaded and not yet removed
                i_date = j[j.index("_")+1:j.index("_")+9]
                imday = int(i_date[-4:])
                if imday < 1216:
                    poly_data[id]['images'][img_count] = dict.fromkeys(band_names)
                    input_filename = os.path.join(data_dir,j)
                    try:
                        with rio.open(input_filename) as ras:
                            ras.blue = ras.read(1)
                    except:
                        print("problem processing image {}. skipping.".format(j))
                        img_count = img_count+1
                        pass
                    else:
                        with rio.open(input_filename) as ras:
                            ras.blue = ras.read(1, masked=True)
                            ras.green = ras.read(2, masked=True)
                            ras.red = ras.read(3, masked=True)
                            if num_bands > 3:
                                ras.nir = ras.read(4, masked=True)
                            elif num_bands == 9:
                                ras.nir = ras.read(7, masked=True)
                                ras.redEdge1 = ras.read(4, masked=True)
                                ras.redEdge2 = ras.read(5, masked=True)
                                ras.redEdge3 = ras.read(6, masked=True)
                                ras.SWIR1 = ras.read(8, masked=True)
                                ras.SWIR2 = ras.read(9, masked=True)
                        ##Get flattened arrays (ordered lists) for all 4 bands
                        bluevals = ma.masked_array(ras.blue, dtype='float64')
                        raspix = ma.MaskedArray.count(bluevals)
                        numpix.append(tuple([input_filename, raspix]))
                        bluevals = bluevals.flatten()
                        bluevals = bluevals.filled(np.NAN) 
                        poly_data[id]['images'][img_count]['blue']=bluevals
                        #PolyPix = np.count_nonzero(~np.isnan(blueVals)) #Already counting the masked array above
                        #print("{} frame pix and {} poly pix in img {}".format(RasPix, PolyPix, imgCount))
                        greenvals = ma.masked_array(ras.green, dtype='float64')
                        greenvals = greenvals.flatten()
                        greenvals = greenvals.filled(np.NAN) 
                        poly_data[id]['images'][img_count]['green']=greenvals
                        redvals = ma.masked_array(ras.red, dtype='float64')
                        redvals = redvals.flatten()
                        redvals = redvals.filled(np.NAN)
                        poly_data[id]['images'][img_count]['red']=redvals
                        if num_bands >3:
                            nirvals = ma.masked_array(ras.nir, dtype='float64')
                            nirvals = nirvals.flatten()
                            nirvals = nirvals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['nir']=nirvals
                        if num_bands == 9:     
                            rededge1vals = ma.masked_array(ras.redEdge1, dtype='float64')
                            rededge1vals = rededge1vals.flatten()
                            rededge1vals = rededge1vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['redEdge1']=rededge1vals
                            rededge2vals = ma.masked_array(ras.redEdge2, dtype='float64')
                            rededge2vals = rededge2vals.flatten()
                            rededge2vals = rededge2vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['redEdge2']=rededge2vals
                            rededge3vals = ma.masked_array(ras.redEdge3, dtype='float64')
                            rededge3vals = rededge3vals.flatten()
                            rededge3vals = rededge3vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['redEdge3']=rededge3vals
                            swir1vals = ma.masked_array(ras.SWIR1, dtype='float64')
                            swir1vals = swir1vals.flatten()
                            swir1vals = swir1vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['SWIR1']=swir1vals
                            swir2vals = ma.masked_array(ras.SWIR2, dtype='float64')
                            swir2vals = swir2vals.flatten()
                            swir2vals = swir2vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['SWIR2']=swir2vals                    
                                                 
                        img_count = img_count+1
                        goodimg_count = goodimg_count + 1
        if goodimg_count == 0:
            blank_ids.append(id)
            print("there are no valid images for field {}".format(id))
        elif goodimg_count > 0:
            ##Get coordinates from image with max unasked pixels (because clouds were masked prior to clipping imagery)
            maxPix = max(numpix, key=operator.itemgetter(1)) 
            poly_data[id]['coords'] = get_pix_coords(maxPix[0])
         
    if len(blank_ids)>0:
        blank_file = open(os.path.join(out_dir,'ids_without_images.txt'),'w')
        for element in blank_ids:
            blank_file.write(str(element) + "\n")
        print("there are {} ids with no images, written to file:{}".format(len(blank_ids),blank_file))

    return poly_data


def pixel_level_calcs(poly_id, poly_dict, out_dir, num_bands, var_path):
    
    #Get list of variables to include in model:
    var = []
    scale = 10000
    with open(var_path, newline='') as f:
        for row in csv.reader(f):
            var.append(row[0])
            
    blue_arrays=[]
    green_arrays=[]
    red_arrays=[]
    if num_bands > 3:
        nir_arrays=[]
    if num_bands == 9:
        rededge1_arrays=[]
        rededge2_arrays=[]
        rededge3_arrays=[]
        swir1_arrays=[]
        swir2_arrays=[]
    
    ##For each image, get pixel values for loaded bands and append to band array holding pixel-values for all images:   
    #poly_data[poly_id] has been passed as poly_dict)
    for key, value in poly_dict['images'].items():
        print("second dict key (image): {}".format(key))
        img_id = key
        blue_series = poly_dict['images'][img_id]['blue']
        blue_arrays.append(blue_series)
        green_series = poly_dict['images'][img_id]['green']
        green_arrays.append(green_series)
        red_series = poly_dict['images'][img_id]['red']
        red_arrays.append(red_series)
        if num_bands > 3:
            nir_series = poly_dict['images'][img_id]['nir']
            nir_arrays.append(nir_series)
        if num_bands == 9:
            rededge1_series=poly_dict['images'][img_id]['redEdge1']
            rededge1_arrays.append(rededge1_series)
            rededge2_series=poly_dict['images'][img_id]['redEdge2']
            rededge2_arrays.append(rededge2_series)
            rededge3_series=poly_dict['images'][img_id]['redEdge3']
            rededge3_arrays.append(rededge3_series)
            swir1_series=poly_dict['images'][img_id]['SWIR1']
            swir1_arrays.append(swir1_series)
            SWIR2_series=poly_dict['images'][img_id]['SWIR2']
            swir2_arrays.append(SWIR2_series)

    ##Convert final array to matrix with one row for each image and one column for each pixel and get summary calcs       
    blue_matrix = pd.DataFrame(blue_arrays)
    blue_matrix = blue_matrix.dropna(axis='columns', how='all')
    green_matrix = pd.DataFrame(green_arrays)
    green_matrix = green_matrix.dropna(axis='columns', how='all')
    red_matrix = pd.DataFrame(red_arrays)
    red_matrix = red_matrix.dropna(axis='columns', how='all')
    if num_bands >3: 
        nir_matrix = pd.DataFrame(nir_arrays)
        nir_matrix = nir_matrix.dropna(axis='columns', how='all')
    if num_bands == 9:
        rededge1_matrix = pd.DataFrame(rededge1_arrays)
        rededge1_matrix = rededge1_matrix.dropna(axis='columns', how='all')
        rededge2_matrix = pd.DataFrame(rededge2_arrays)
        rededge2_matrix = rededge2_matrix.dropna(axis='columns', how='all')
        rededge3_matrix = pd.DataFrame(rededge3_arrays)
        rededge3_matrix = rededge3_matrix.dropna(axis='columns', how='all')
        swir1_matrix = pd.DataFrame(swir1_arrays)
        swir1_matrix = swir1_matrix.dropna(axis='columns', how='all')
        swir2_matrix = pd.DataFrame(swir2_arrays)
        swir2_matrix = swir2_matrix.dropna(axis='columns', how='all')
    
    ##Calculate indices
    ##Vegetation indices
    if 'ndviAvg' in var:
        ndvi_matrix = scale * (nir_matrix - red_matrix)/(nir_matrix + red_matrix)
    if 'evi2Avg' in var:
        evi2_matrix = 2.5 * scale * ((nir_matrix/scale - red_matrix/scale) / (nir_matrix/scale + 1.0 + 2.4 * red_matrix/scale))
    if 'saviAvg' in var:
        lfactor = .5 #(0-1, 0=very green, 1=very arid. .5 most common. Some use negative vals for arid env)
        savi_matrix = scale * (1 + lfactor) * ((nir_matrix/scale - red_matrix/scale) / (nir_matrix/scale + red_matrix/scale + lfactor))
    if 'msaviAvg' in var:
        msavi_matrix = scale/2 * (2 * nir_matrix/scale + 1) - ((2 * nir_matrix/scale + 1)**2 - 8*(nir_matrix/scale - red_matrix/scale))**1/2
    ##Burn indices
    if 'bareSoilAvg' in var:
        baresoil_matrix = scale * ((nir_matrix/scale + green_matrix/scale) - (red_matrix + blue_matrix))/(nir_matrix + green_matrix + red_matrix + blue_matrix)*100+100
    if 'srAvg' in var:
        sr_matrix = scale * nir_matrix / red_matrix
    if 'baiAvg' in var: #lower means more charred
        bai_matrix = scale/((.06-(nir_matrix/scale))**2+(.1-(red_matrix/scale))**2)
    if 'ciAvg' in var:  #lower means more charred
        vissum = (blue_matrix/scale + green_matrix/scale + red_matrix/scale)
        bg = abs(blue_matrix/scale - green_matrix/scale)
        br = abs(blue_matrix/scale - red_matrix/scale)
        rg = abs(red_matrix/scale - green_matrix/scale)
        maxdiff1 = np.maximum(bg, br)
        maxdiff = np.maximum(maxdiff1, rg)
        ci_matrix = scale * (vissum + (maxdiff*15))
    ##Moisture indices:
    if 'ndmi_matrix' in var:
        ndmi_matrix = scale * (nir_matrix - green_matrix) / ((nir_matrix + green_matrix) + 1e-9)
         
    if num_bands == 9:
        if 'NBRAvg' in var:
            nbr_matrix = scale * (nir_matrix - swir2_matrix)/(nir_matrix + swir2_matrix)
        if 'NBR2Avg' in var:
            nbr2_matrix = scale * (swir1_matrix - swir2_matrix)/(swir1_matrix + swir2_matrix)
        if 'MIRBIAvg' in var: mirbi_matrix = scale * (10*swir2_matrix/scale - 9.8*swir1_matrix/scale + 2)
        if 'baiSAvg' in var:
            bais_matrix = scale * (1 - np.math.sqrt((rededge2_matrix/scale * rededge3_matrix/scale * nir_matrix/scale) / red_matrix/scale)) * (((swir2_matrix/scale - nir_matrix/scale) / np.math.sqrt(swir2_matrix/scale + nir_matrix/scale)) + 1)
        if 'BurnScarAvg' in var:
            m = 2 #(try 1,2,4,6) until seperability is maximized
            burnbare_ratio = (swir2_matrix - red_matrix)/ (swir2_matrix + red_matrix)
            background = ((green_matrix/scale)**m + (red_matrix/scale)**m + (nir_matrix/scale)**m)
            burnscar_matrix = scale * (burnbare_ratio * 1/background)
        #if 'TCWetAvg' in var: #Tasseled Cap coefficients only available for TOA Sentinel imagery
        if 'ndwi_matrix' in var:
            ndwi_matrix = scale * (nir_matrix - swir1_matrix) / ((nir_matrix + swir1_matrix) + 1e-9)
         
    pixel_data = {}
    drop_pixels = [] #TODO: drop these from dataframe later
    ##Get pixel-level summary calcs
    pixels = blue_matrix.columns.tolist()
    for p in range(len(pixels)):
        pix_num = pixels[p]
        #print(pix_num)
        pix_id = str(poly_id)+"_"+str(pix_num)
        pixel_data[pix_id] = {}
        if pix_num not in poly_dict['coords']:
            drop_pixels.append(pix_id)
            pass
        else:
            ###Get coordinates if geometry changes, otherwise do not need to do this every time
            pixel_data[pix_id]["CoordX"]= poly_dict['coords'][pix_num]['Xcoord']
            pixel_data[pix_id]["CoordY"]= poly_dict['coords'][pix_num]['Ycoord']
            #print("X, Y coords are: {}, {}".format(pixel_data[pix_id]["CoordX"], pixel_data[pix_id]["CoordY"]))
            
            ###pixel_data={'pixel': pix_id{}}
            num_images = np.count_nonzero(~np.isnan(blue_matrix[pix_num]))
            pixel_data[pix_id]["imgCount"]=num_images
            if num_images > 3:  #Use 3 with BASMA, 4 with regular Sentinel, 13 with Planet
                if 'blueAvg' in var:
                    if num_bands == 3:
                        pixel_data[pix_id]["blueAvg"]=blue_matrix[pix_num].mean()
                    else:
                        pixel_data[pix_id]["blueAvg"]=int(blue_matrix[pix_num].mean())
                if 'blueMax' in var:
                    if num_bands == 3:
                        pixel_data[pix_id]["blueMax"]=blue_matrix[pix_num].max()
                    else:
                        pixel_data[pix_id]["blueMax"]=int(blue_matrix[pix_num].max())
                if 'blueMin' in var:
                    pixel_data[pix_id]["blueMin"]=int(blue_matrix[pix_num].min())
                if 'blueRng' in var:
                    pixel_data[pix_id]["blueRng"]=int(blue_matrix[pix_num].max())-int(blue_matrix[pix_num].min())
                if 'blueStdv' in var:
                    if num_bands == 3:
                        pixel_data[pix_id]["blueStdv"]=blue_matrix[pix_num].std()
                    else:
                        pixel_data[pix_id]["blueStdv"]=int(blue_matrix[pix_num].std())
                if 'greenAvg' in var:
                    if num_bands == 3:
                        pixel_data[pix_id]["greenAvg"]=green_matrix[pix_num].mean()
                    else:
                        pixel_data[pix_id]["greenAvg"]=int(green_matrix[pix_num].mean())
                if 'greenMax' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["greenMax"]=green_matrix[pix_num].max()
                    else:
                        pixel_data[pix_id]["greenMax"]=int(green_matrix[pix_num].max())
                if 'greenMin' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["greenMin"]=green_matrix[pix_num].min()
                    else:
                        pixel_data[pix_id]["greenMin"]=int(green_matrix[pix_num].min())
                if 'greenRng' in var:
                    pixel_data[pix_id]["greenRng"]=int(green_matrix[pix_num].max())-int(green_matrix[pix_num].min())
                if 'greenStdv' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["greenStdv"]=green_matrix[pix_num].std()
                    else:
                        pixel_data[pix_id]["greenStdv"]=int(green_matrix[pix_num].std())
                if 'redAvg' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["redAvg"]=red_matrix[pix_num].mean()
                    else:
                        pixel_data[pix_id]["redAvg"]=int(red_matrix[pix_num].mean())
                if 'redMax' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["redMax"]=red_matrix[pix_num].max()
                    else:
                        pixel_data[pix_id]["redMax"]=int(red_matrix[pix_num].max())
                    #print("redMax for pix {} is {}".format(pix_id, pixel_data[pix_id]["redMax"]))
                if 'redMin' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["redMin"]=red_matrix[pix_num].min()
                    else:
                        pixel_data[pix_id]["redMin"]=int(red_matrix[pix_num].min())
                if 'redRng' in var:
                    pixel_data[pix_id]["redRng"]=int(red_matrix[pix_num].max())-int(red_matrix[pix_num].min())
                if 'redStdv' in var:
                    if num_bands == 3:   
                        pixel_data[pix_id]["redStdv"]=red_matrix[pix_num].std()
                    else:
                        pixel_data[pix_id]["redStdv"]=int(red_matrix[pix_num].std())
                if 'red80' in var:
                    pixel_data[pix_id]["red80"]=int(np.nanpercentile(red_matrix[pix_num],80))
                if 'red20' in var:
                    pixel_data[pix_id]["red20"]=int(np.nanpercentile(red_matrix[pix_num],20))
                if 'nirAvg' in var:
                    pixel_data[pix_id]["nirAvg"]=int(nir_matrix[pix_num].mean())
                if 'nirMax' in var:
                    pixel_data[pix_id]["nirMax"]=int(nir_matrix[pix_num].max())
                if 'nirMin' in var:
                    pixel_data[pix_id]["nirMin"]=int(nir_matrix[pix_num].min())
                if 'nir20' in var:
                    pixel_data[pix_id]["nir20"]=np.nanpercentile(nir_matrix[pix_num],20)
                if 'nirRng' in var:
                    pixel_data[pix_id]["nirRng"]=int(nir_matrix[pix_num].max())-int(nir_matrix[pix_num].min())
                if 'nirStdv' in var:
                    pixel_data[pix_id]["nirStdv"]=int(nir_matrix[pix_num].std())
                if 'SWIR2Avg' in var:
                    pixel_data[pix_id]["SWIR2Avg"]=int(swir2_matrix[pix_num].mean())
                if 'SWIR2Max' in var:
                    pixel_data[pix_id]["SWIR2Max"]=int(swir2_matrix[pix_num].max())
                if 'SWIR2Min' in var:
                    pixel_data[pix_id]["SWIR2Min"]=int(swir2_matrix[pix_num].min())
                if 'SWIR1Avg' in var:
                    pixel_data[pix_id]["SWIR1Avg"]=int(swir2_matrix[pix_num].mean())
                if 'SWIR1Max' in var:
                    pixel_data[pix_id]["SWIR1Max"]=int(swir2_matrix[pix_num].max())
                if 'SWIR1Min' in var:
                    pixel_data[pix_id]["SWIR1Min"]=int(swir2_matrix[pix_num].min())
                if 'ndviAvg' in var:
                    pixel_data[pix_id]["ndviAvg"]=ndvi_matrix[pix_num].mean()
                if 'ndviMax' in var:
                    pixel_data[pix_id]["ndviMax"]=ndvi_matrix[pix_num].max()
                if 'ndviMin' in var:
                    pixel_data[pix_id]["ndviMin"]=ndvi_matrix[pix_num].min()
                if 'ndviRng' in var:    
                    pixel_data[pix_id]["ndviRng"]=ndvi_matrix[pix_num].max()-ndvi_matrix[pix_num].min()
                if 'ndviStdv' in var:
                    pixel_data[pix_id]["ndviStdv"]=ndvi_matrix[pix_num].std()
                if 'ndvi25' in var:
                    pixel_data[pix_id]["ndvi25"]=np.nanpercentile(ndvi_matrix[pix_num],25)
                if 'ndvi10' in var:
                    pixel_data[pix_id]["ndvi10"]=np.nanpercentile(ndvi_matrix[pix_num],10)
                if 'NBRAvg' in var:
                    pixel_data[pix_id]["NBRAvg"]=nbr_matrix[pix_num].mean()
                if 'NBRMax' in var:
                    pixel_data[pix_id]["NBRMax"]=nbr_matrix[pix_num].max()
                if 'NBRMin' in var:
                    pixel_data[pix_id]["NBRMin"]=nbr_matrix[pix_num].min()
                if 'NBR2Avg' in var:
                    pixel_data[pix_id]["NBR2Avg"]=nbr_matrix[pix_num].mean()
                if 'NBR2Max' in var:
                    pixel_data[pix_id]["NBR2Max"]=nbr_matrix[pix_num].max()
                if 'NBR2Min' in var:
                    pixel_data[pix_id]["NBR2Min"]=nbr_matrix[pix_num].min()
                if 'bareSoilAvg' in var:
                    pixel_data[pix_id]["bareSoilAvg"]=int(baresoil_matrix[pix_num].mean())
                if 'bareSoilMax' in var:
                    pixel_data[pix_id]["bareSoilMax"]=int(baresoil_matrix[pix_num].max())
                if 'bareSoilMin' in var:
                    pixel_data[pix_id]["bareSoilMin"]=int(baresoil_matrix[pix_num].min())
                if 'CIAvg' in var:
                    pixel_data[pix_id]["CIAvg"]=int(ci_matrix[pix_num].mean())
                if 'CIMin' in var:
                    pixel_data[pix_id]["CIMin"]=int(ci_matrix[pix_num].min())
                if 'CI80' in var:
                    pixel_data[pix_id]["CI80"]=int(np.nanpercentile(ci_matrix[pix_num],80))
                if 'CIRng' in var:
                    pixel_data[pix_id]["CIRng"]=int(ci_matrix[pix_num].max())-int(ci_matrix[pix_num].min())
                if 'baiAvg' in var:
                    pixel_data[pix_id]["baiAvg"]=int(bai_matrix[pix_num].mean())
                if 'baiMax' in var:
                    pixel_data[pix_id]["baiMax"]=int(bai_matrix[pix_num].max())
                if 'baiMin' in var:
                    pixel_data[pix_id]["baiMin"]=int(bai_matrix[pix_num].min())
                if 'baiSAvg' in var:
                    pixel_data[pix_id]["baiSAvg"]=int(bais_matrix[pix_num].mean())
                if 'baiSMax' in var:
                    pixel_data[pix_id]["baiSMax"]=int(bais_matrix[pix_num].max())
                if 'baiSMin' in var:
                    pixel_data[pix_id]["baiSMin"]=int(bais_matrix[pix_num].min())
                if 'bareSoilRng' in var:
                    pixel_data[pix_id]["bareSoilRng"]=int(baresoil_matrix[pix_num].max())-int(baresoil_matrix[pix_num].min())
                if 'bareSoilStdv' in var:
                    pixel_data[pix_id]["bareSoilStdv"]=int(baresoil_matrix[pix_num].std())
                if 'bareSoil20' in var:
                    pixel_data[pix_id]["bareSoil20"]=int(np.nanpercentile(baresoil_matrix[pix_num],20))
                if 'bareSoil10' in var:
                    pixel_data[pix_id]["bareSoil10"]=int(np.nanpercentile(baresoil_matrix[pix_num],10))
                if 'srAvg' in var:
                    pixel_data[pix_id]["srAvg"]=sr_matrix[pix_num].mean()
                if 'srMax' in var:    
                    pixel_data[pix_id]["srMax"]=sr_matrix[pix_num].max()
                if 'srMin' in var:
                    pixel_data[pix_id]["srMin"]=sr_matrix[pix_num].min()
                if 'srRng' in var:
                    pixel_data[pix_id]["srRng"]=sr_matrix[pix_num].max()-sr_matrix[pix_num].min()
                if 'srStdv' in var:
                    pixel_data[pix_id]["srStdv"]=sr_matrix[pix_num].std()
                if 'sr25' in var:
                    pixel_data[pix_id]["sr25"]=np.nanpercentile(sr_matrix[pix_num],25)
                if 'sr10' in var:
                    pixel_data[pix_id]["sr10"]=np.nanpercentile(sr_matrix[pix_num],10)
                if 'MIRBIAvg' in var:
                    pixel_data[pix_id]["MIRBIAvg"]=mirbi_matrix[pix_num].mean()
                if 'MIRBIMax' in var:
                    pixel_data[pix_id]["MIRBIMax"]=mirbi_matrix[pix_num].max()
                if 'MIRBI80' in var:
                    pixel_data[pix_id]["MIRBI80"]=np.nanpercentile(mirbi_matrix[pix_num],80)
                if 'BurnScarAvg' in var:
                    pixel_data[pix_id]["BurnScarAvg"]=burnscar_matrix[pix_num].mean()
                if 'BurnScarMax' in var:
                    pixel_data[pix_id]["BurnScarMax"]=burnscar_matrix[pix_num].max()
                if 'BurnScar80' in var:
                    pixel_data[pix_id]["BurnScar80"]=np.nanpercentile(burnscar_matrix[pix_num],80)
                if 'redEdge3Avg' in var:
                    pixel_data[pix_id]["redEdge3Avg"]=rededge3_matrix[pix_num].mean()
                if 'redEdge3Min' in var:
                    pixel_data[pix_id]["redEdge3Min"]=rededge3_matrix[pix_num].min()
                if 'redEdge320' in var:
                    pixel_data[pix_id]["redEdge320"]=np.nanpercentile(rededge3_matrix[pix_num],20)
                if 'evi2Avg' in var:
                    pixel_data[pix_id]["evi2Avg"]=evi2_matrix[pix_num].mean()
                if 'evi2Max' in var:
                    pixel_data[pix_id]["evi2Max"]=evi2_matrix[pix_num].max()
                if 'evi2Min' in var:
                    pixel_data[pix_id]["evi2Min"]=evi2_matrix[pix_num].min()
                if 'saviAvg' in var:
                    pixel_data[pix_id]["saviAvg"]=savi_matrix[pix_num].mean()
                if 'saviMax' in var:
                    pixel_data[pix_id]["saviMax"]=savi_matrix[pix_num].max()
                if 'saviMin' in var:
                    pixel_data[pix_id]["saviMin"]=savi_matrix[pix_num].min()
                if 'msaviAvg' in var:
                    pixel_data[pix_id]["msaviAvg"]=msavi_matrix[pix_num].mean()
                if 'msaviMax' in var:
                    pixel_data[pix_id]["msaviMax"]=msavi_matrix[pix_num].max()
                if 'msaviMin' in var:
                    pixel_data[pix_id]["msaviMin"]=msavi_matrix[pix_num].min()
                
        
                ## Get Max spike/drop in time series vals (filtered for sustained change to control for clouds) 
                if 'ndviDrop2' in var:
                    pixel_data[pix_id]["ndviDrop2"]= max_drop(ndvi_matrix[pix_num], pixel_data[pix_id]["ndviAvg"],2)
                    #print("NDVIDrop for pix {} is {}".format(pix_id, pixel_data[pix_id]["ndviDrop"]))
                if 'bareSoilDrop2' in var:
                    pixel_data[pix_id]["bareSoilDrop0"]= max_drop(baresoil_matrix[pix_num], pixel_data[pix_id]["bareSoilAvg"],2)
                if 'bareSoilDrop1' in var:
                    pixel_data[pix_id]["bareSoilDrop1"]= max_drop(baresoil_matrix[pix_num], pixel_data[pix_id]["bareSoilAvg"],1)
                if 'CIDrop1' in var:  #CI drop indicates no char to char
                    pixel_data[pix_id]["CIDrop1"]= max_drop(ci_matrix[pix_num], pixel_data[pix_id]["CIAvg"],1)
                if 'CIDrop2' in var:
                    pixel_data[pix_id]["CIDrop2"]= max_drop(ci_matrix[pix_num], pixel_data[pix_id]["CIAvg"],2)
                if 'CIDrop0' in var:
                    pixel_data[pix_id]["CIDrop0"]= max_drop(ci_matrix[pix_num], pixel_data[pix_id]["CIAvg"],0)
                if 'CISpike1' in var: #CI Spike indicates char to no char
                    pixel_data[pix_id]["CISpike1"]= max_spike(ci_matrix[pix_num], pixel_data[pix_id]["CIAvg"],1)
                if 'CISpike2' in var:    
                    pixel_data[pix_id]["CISpike2"]= max_spike(ci_matrix[pix_num], pixel_data[pix_id]["CIAvg"],2)
                if 'CISpike0' in var:
                    pixel_data[pix_id]["CISpike0"]= max_spike(ci_matrix[pix_num], pixel_data[pix_id]["CIAvg"],0)
                if 'baiDrop1' in var:
                    pixel_data[pix_id]["baiDrop1"]= max_drop(bai_matrix[pix_num], pixel_data[pix_id]["baiAvg"],1)
                if 'baiDrop2' in var:
                    pixel_data[pix_id]["baiDrop2"]= max_drop(bai_matrix[pix_num], pixel_data[pix_id]["baiAvg"],2)
                if 'baiDrop0' in var:
                    pixel_data[pix_id]["baiDrop0"]= max_drop(bai_matrix[pix_num], pixel_data[pix_id]["baiAvg"],0)
                if 'baiSDrop1' in var:
                    pixel_data[pix_id]["baiSDrop1"]= max_drop(bais_matrix[pix_num], pixel_data[pix_id]["baiSAvg"],1)
                if 'baiSDrop2' in var:
                    pixel_data[pix_id]["baiSDrop2"]= max_drop(bais_matrix[pix_num], pixel_data[pix_id]["baiSAvg"],2)
                if 'baiSDrop0' in var:
                    pixel_data[pix_id]["baiSDrop0"]= max_drop(bais_matrix[pix_num], pixel_data[pix_id]["baiSAvg"],0)
                if 'srDrop2' in var:
                    pixel_data[pix_id]["srDrop2"]= max_drop(sr_matrix[pix_num], pixel_data[pix_id]["srAvg"],2)
                if 'srDrop1' in var:
                    pixel_data[pix_id]["srDrop1"]= max_drop(sr_matrix[pix_num], pixel_data[pix_id]["srAvg"],1)
                if 'srDrop0' in var:
                    pixel_data[pix_id]["srDrop0"]= max_drop(sr_matrix[pix_num], pixel_data[pix_id]["srAvg"],0)
                if 'nirDrop1' in var:
                    pixel_data[pix_id]["nirDrop1"]= max_drop(nir_matrix[pix_num], pixel_data[pix_id]["nirAvg"],1)
                if 'nirDrop2' in var:
                    pixel_data[pix_id]["nirDrop2"]= max_drop(nir_matrix[pix_num], pixel_data[pix_id]["nirAvg"],2)
                if 'nirDrop0' in var:
                    pixel_data[pix_id]["nirDrop0"]= max_drop(nir_matrix[pix_num], pixel_data[pix_id]["nirAvg"],0)
                if 'redDrop1' in var:
                    pixel_data[pix_id]["redDrop1"]= max_drop(red_matrix[pix_num], pixel_data[pix_id]["redAvg"],1)
                    #print("redDrop for pix {} of poly {} is {}".format(poly_id, pix_id, pixel_data[pix_id]["redDrop"]))
                if 'redDrop0' in var:
                    pixel_data[pix_id]["redDrop0"]= max_drop(red_matrix[pix_num], pixel_data[pix_id]["redAvg"],0)
                if 'redDrop2' in var:
                    pixel_data[pix_id]["redDrop1"]= max_drop(red_matrix[pix_num], pixel_data[pix_id]["redAvg"],2)
                if 'redSpike0' in var:
                    pixel_data[pix_id]["redSpike0"]= max_spike(red_matrix[pix_num], pixel_data[pix_id]["redAvg"],0)
                if 'greenSpike0' in var:
                    pixel_data[pix_id]["greenSpike0"]= max_spike(green_matrix[pix_num], pixel_data[pix_id]["greenAvg"],0)
                if 'greenDrop0' in var:
                    pixel_data[pix_id]["greenDrop0"]= max_drop(green_matrix[pix_num], pixel_data[pix_id]["greenAvg"],0)
                if 'redEdge3Drop1' in var:
                    pixel_data[pix_id]["redEdge3Drop1"]= max_drop(rededge3_matrix[pix_num], pixel_data[pix_id]["redEdge3Avg"],1)
                if 'redEdge3Drop0' in var:
                    pixel_data[pix_id]["redEdge3Drop0"]= max_drop(rededge3_matrix[pix_num], pixel_data[pix_id]["redEdge3Avg"],0)
                if 'blueSpike2' in var: 
                    pixel_data[pix_id]["blueSpike2"]= max_spike(blue_matrix[pix_num], pixel_data[pix_id]["blueAvg"],2)
                if 'blueSpike1' in var:
                    pixel_data[pix_id]["blueSpike1"]= max_spike(blue_matrix[pix_num], pixel_data[pix_id]["blueAvg"],1)
                if 'blueSpike0' in var:
                    pixel_data[pix_id]["blueSpike0"]= max_spike(blue_matrix[pix_num], pixel_data[pix_id]["blueAvg"],0)
                if 'MIRBISpike1' in var:
                    pixel_data[pix_id]["MIRBISpike1"]= max_spike(mirbi_matrix[pix_num], pixel_data[pix_id]["MIRBIAvg"],1)
                if 'MIRBISpike0' in var:
                    pixel_data[pix_id]["MIRBISpike0"]= max_spike(mirbi_matrix[pix_num], pixel_data[pix_id]["MIRBIAvg"],0)
                if 'NBRDrop1' in var:
                    pixel_data[pix_id]["NBRDrop1"]= max_drop(nbr_matrix[pix_num], pixel_data[pix_id]["NBRAvg"],1)
                if 'NBRDrop0' in var:
                    pixel_data[pix_id]["NBRDrop0"]= max_drop(nbr_matrix[pix_num], pixel_data[pix_id]["NBRAvg"],0)
                if 'NBR2Drop1' in var:
                    pixel_data[pix_id]["NBR2Drop1"]= max_drop(nbr2_matrix[pix_num], pixel_data[pix_id]["NBR2Avg"],1)
                if 'NBR2Drop0' in var:
                    pixel_data[pix_id]["NBR2Drop0"]= max_drop(nbr2_matrix[pix_num], pixel_data[pix_id]["NBR2Avg"],0)
                if 'BurnScarSpike1' in var:
                    pixel_data[pix_id]["BurnScarSpike1"]= max_spike(burnscar_matrix[pix_num], pixel_data[pix_id]["BurnScarAvg"],1)
                if 'BurnScarSpike0' in var:
                    pixel_data[pix_id]["BurnScarSpike0"]= max_spike(burnscar_matrix[pix_num], pixel_data[pix_id]["BurnScarAvg"],0)
                if 'SWIR1Drop1' in var:
                    pixel_data[pix_id]["SWIR1Drop1"]= max_drop(swir1_matrix[pix_num], pixel_data[pix_id]["SWIR1Avg"],1)
                if 'SWIR1Drop0' in var:
                    pixel_data[pix_id]["SWIR1Drop0"]= max_drop(swir1_matrix[pix_num], pixel_data[pix_id]["SWIR1Avg"],0)
                if 'SWIR2Drop1' in var:
                    pixel_data[pix_id]["SWIR2Drop1"]= max_drop(swir2_matrix[pix_num], pixel_data[pix_id]["SWIR2Avg"],1)
                if 'SWIR2Drop0' in var:
                    pixel_data[pix_id]["SWIR2Drop0"]= max_drop(swir2_matrix[pix_num], pixel_data[pix_id]["SWIR2Avg"],0)
            else: 
                drop_pixels.append(pix_id)

    if len(drop_pixels)>0:
        drop_file = open(os.path.join(out_dir,'pixels_with_fewer_than_5_images.txt'),'w')
        for element in drop_pixels:
            drop_file.write(str(element) + "\n")
        print("there are {} pixels with <5 images, or not in coord index, written to file:{}".format(len(drop_pixels),drop_file))
        
    return pixel_data


def get_all_pixel_calcs(field_list, data_dir, out_dir, num_bands, var_path):
    pixelsets = []
    poly_data = poly_data_to_dict(field_list, data_dir, out_dir, num_bands)

    for key, value in poly_data.items():
        print("first dict key (polygon): {}".format(key))
        poly_id = key
        pixel_data = pixel_level_calcs(poly_id, poly_data[poly_id], out_dir, num_bands, var_path)
        pixelset = pd.DataFrame(pixel_data).T
        #pd.DataFrame.to_csv(pixelset, os.path.join(out_dir,'Pixels_'+str(poly_id)+'.csv'), sep=',', na_rep='NaN', index=True)
        pixelsets.append(pixelset)
    
    pixel_sheet =  pd.concat(pixelsets, ignore_index=False)
    #pd.DataFrame.to_csv(pixel_sheet, os.path.join(out_dir,'AllPixels_Training.csv'), sep=',', na_rep='NaN', index=True)
    if num_bands == 3:
        pd.DataFrame.to_csv(pixel_sheet, os.path.join(out_dir,'V4_pixelData_BASMA.csv'), sep=',', na_rep='NaN', index=True, index_label='pixel_id')
    elif num_bands == 4:
        pd.DataFrame.to_csv(pixel_sheet, os.path.join(out_dir,'V4_pixelData_Planet.csv'), sep=',', na_rep='NaN', index=True, index_label='pixel_id')
    elif num_bands == 9:
        pd.DataFrame.to_csv(pixel_sheet, os.path.join(out_dir,'V4_pixelData_Sentinel.csv'), sep=',', na_rep='NaN', index=True, index_label='pixel_id')
    
    print("done")
    return None


def get_coordinate_df (pixel_df, out_dir):
    '''
    Separate off coordinates from Pixel Dataframe to input into ArcGIS for other geographic variables 
    (only need to do once). Note that Shapefiles are limited to ~2GB and these files exceed that, so import
    into feature class within an ArcGIS Geodatabase to process geographic data
    or change this so that it writes as a GeoJson and manipulate from there.
    '''
    all_pixels_c = pd.read_csv(pixel_df, index_col=False)
    #ALL_pixelsC.rename(columns={'Unnamed: 0':'pixel_id'}, inplace=True)
    #print(ALL_pixelsC.columns)

    c_variables = ['pixel_id','CoordX','CoordY','imgCount']
    all_pixels_c1 = all_pixels_c[all_pixels_c.columns.intersection(c_variables)]
    print(len(all_pixels_c1))
    all_pixels_c1 = all_pixels_c1.dropna(how='any',axis=0)
    ##TODO: Find source of NAs. Should they just be dropped like this?
    numrows = len(all_pixels_c1)
    print(numrows)
    
    ##X and Y coords being written as strings and won't import into ArcGIS. Convert to numbers here
    ##  note still being read as strings by ArcGIS. I converted from string to int within ArcGIS instead.
    #all_pixels_c1['xCoord']=all_pixels_c1['CoordX'].astype(float)
    #all_pixels_c1['yCoord']=all_pixels_c1['CoordY'].astype(float)
    
    unique_ids = len(all_pixels_c1.pixel_id.unique())
    if numrows == unique_ids:
        print('pixel IDs and row numbers match. Printing result.')
        pd.DataFrame.to_csv(all_pixels_c1, os.path.join(out_dir,'ALLpix_Geographic.csv'), sep=',', na_rep='NaN', index=True)             
    else:
        print("ERROR: {} unique pixel IDs for {} rows".format(numrows, unique_ids))
        all_pixels_c2 = all_pixels_c1.drop_duplicates(subset=['pixel_id'], keep='first')
        pd.DataFrame.to_csv(all_pixels_c2, os.path.join(out_dir,'ALLpix_Geographic.csv'), sep=',', na_rep='NaN', index=True)


def add_geographic_info (pixeldf_path, pixel_geog, out_dir):
    '''One-time pixel-level dataset prep 
    (combines fields from ArcGIS analysis (separate process) and adds to PixelCalcs dataset)
    'FieldSize' is area of the field in square m
    'FieldNbs' = number of (8) neighboring pixels that are within the field
    'border' = 1 if any of 8 neighboring pixels are within the field
    '''
    ###Get the main pixel dataframe (output from pixel_calcs) 
    #pixel_df1 = pd.read_csv(pixeldf_path, index_col=[0])
    pixel_df1 = pd.read_csv(pixeldf_path)
    pixel_df1['pixel_id'] = pixel_df1['pixel_id'].apply(str)
    print(len(pixel_df1))
    ###Add 'FieldSize', 'FieldNbs' & 'border' fields based on separate ArcGIS analysis
    pixel_df2 = pd.read_csv(pixel_geog, usecols=['pixel_id','FieldSize','FieldNbs','border'])
    pixel_df2['pixel_id'] = pixel_df2['pixel_id'].apply(str)
    print(len(pixel_df2))
    ###merge on Unnamed: 0 = pixid
    pixel_df3 = pixel_df1.merge(pixel_df2, how='inner', left_on='pixel_id', right_on='pixel_id')
    #pixel_df3 = pixel_df3.drop('pixel_id', axis=1)
    print('{} unique ids for {} rows'.format(len(pixel_df3.pixel_id.unique()),len(pixel_df3)))
    ###Save as final pixel dataframe to be used in analyses
    pd.DataFrame.to_csv(pixel_df3, os.path.join(out_dir,'pixelData_wGeog_V2.csv'), sep=',', na_rep='NaN', index=True)
    return pixel_df3


##need this to include percentiles in aggregate field calcs because lambda doesn't work in combo with other named functions:
## see: https://github.com/pandas-dev/pandas/issues/28467 & https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function/20233047

#I DON"T THINK IT IS ACTUALLY USING THIS; must be a built-in function now

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_


def get_aggregate_field_calcs (pixeldf_path, label_source, out_dir):
    pixeldf = pd.read_csv(pixeldf_path)
    pixeldf['field_id'] = pixeldf['pixel_id'].str[:10]
    print(len(pixeldf))

    ###Add field-level variables, 
    field_df0 = pixeldf.groupby(['field_id']).agg(imgcount_avg=pd.NamedAgg(column='imgCount', aggfunc='mean'),
                                     FieldSize = pd.NamedAgg(column='FieldSize', aggfunc='mean'))
    ### calculations for border pixels:                                          
    field_df1 = pixeldf[pixeldf['border'] == 1].groupby(['field_id']).agg(
                                    b_Pixcount = pd.NamedAgg(column='border', aggfunc='sum'),
                                    b_nir_drop_avg = pd.NamedAgg(column='nirDrop', aggfunc='mean'),
                                    b_nir_drop_10 = pd.NamedAgg(column='nirDrop', aggfunc=percentile(.1)),
                                    b_nir_drop_25 = pd.NamedAgg(column='nirDrop', aggfunc=percentile(.25)),
                                    b_bsi_drop_avg = pd.NamedAgg(column='bsiDrop', aggfunc='mean'),
                                    b_bsi_drop_10 = pd.NamedAgg(column='bsiDrop', aggfunc=percentile(.10)),
                                    b_bsi_drop_25 = pd.NamedAgg(column='bsiDrop', aggfunc=percentile(.25)),
                                    b_ndvi_drop_avg = pd.NamedAgg(column='ndviDrop', aggfunc='mean'),
                                    b_ndvi_drop_10 = pd.NamedAgg(column='ndviDrop', aggfunc=percentile(.10)),
                                    b_ndvi_drop_25 = pd.NamedAgg(column='ndviDrop', aggfunc=percentile(.25)),
                                    b_red_spike_avg = pd.NamedAgg(column='redSpike', aggfunc='mean'),
                                    b_red_spike_90 = pd.NamedAgg(column='redSpike', aggfunc=percentile(.90)),
                                    b_red_spike_75 = pd.NamedAgg(column='redSpike', aggfunc=percentile(.75)),
                                    b_blue_spike_avg = pd.NamedAgg(column='blueSpike', aggfunc='mean'),
                                    b_sr_drop_avg = pd.NamedAgg(column='srDrop', aggfunc='mean'),
                                    b_sr_drop_10 = pd.NamedAgg(column='srDrop', aggfunc=percentile(.10)),
                                    b_sr_drop_25 = pd.NamedAgg(column='srDrop', aggfunc=percentile(.25)),
                                    b_blue_avg = pd.NamedAgg(column='blueAvg', aggfunc='mean'),
                                    b_red_avg = pd.NamedAgg(column='redAvg', aggfunc='mean'),
                                    b_nir_avg = pd.NamedAgg(column='nirAvg', aggfunc='mean'),
                                    b_ndvi_avg = pd.NamedAgg(column='ndviAvg', aggfunc='mean'),
                                    b_bsi_avg = pd.NamedAgg(column='bsiAvg', aggfunc='mean'),
                                    b_sr_avg = pd.NamedAgg(column='srAvg', aggfunc='mean'),
                                    b_blue_stdv = pd.NamedAgg(column='blueAvg', aggfunc='std'),
                                    b_red_stdv = pd.NamedAgg(column='redAvg', aggfunc='std'),
                                    b_nir_stdv = pd.NamedAgg(column='nirAvg', aggfunc='std'),
                                    b_ndvi_stdv = pd.NamedAgg(column='ndviAvg', aggfunc='std'),
                                    b_bsi_stdv = pd.NamedAgg(column='bsiAvg', aggfunc='std'),
                                    b_sr_stdv = pd.NamedAgg(column='srAvg', aggfunc='std'))
    ### calculations for interior pixels: 
    field_df2 = pixeldf[pixeldf['border'] == 0].groupby(['field_id']).agg(
                                    i_ndvi_drop_avg = pd.NamedAgg(column='ndviDrop', aggfunc='mean'),
                                    i_ndvi_drop_med = pd.NamedAgg(column='ndviDrop', aggfunc='median'),
                                    i_ndvi_drop_min = pd.NamedAgg(column='ndviDrop', aggfunc='min'),
                                    i_ndvi_drop_stdv = pd.NamedAgg(column='ndviDrop', aggfunc='std'),
                                    i_ndvi_drop_10 = pd.NamedAgg(column='ndviDrop', aggfunc=percentile(.10)),
                                    i_ndvi_drop_25 = pd.NamedAgg(column='ndviDrop', aggfunc=percentile(.25)),
                                    i_bsi_drop_avg = pd.NamedAgg(column='bsiDrop', aggfunc='mean'),
                                    i_bsi_drop_med = pd.NamedAgg(column='bsiDrop', aggfunc='median'),
                                    i_bsi_drop_min = pd.NamedAgg(column='bsiDrop', aggfunc='min'),
                                    i_bsi_drop_stdv = pd.NamedAgg(column='bsiDrop', aggfunc='std'),
                                    i_bsi_drop_10 = pd.NamedAgg(column='bsiDrop', aggfunc=percentile(.10)),
                                    i_bsi_drop_25 = pd.NamedAgg(column='bsiDrop', aggfunc=percentile(.25)),
                                    i_nir_drop_avg = pd.NamedAgg(column='nirDrop', aggfunc='mean'),
                                    i_nir_drop_med = pd.NamedAgg(column='nirDrop', aggfunc='median'),
                                    i_nir_drop_min = pd.NamedAgg(column='nirDrop', aggfunc='min'),
                                    i_nir_drop_stdv = pd.NamedAgg(column='nirDrop', aggfunc='std'),
                                    i_nir_drop_10 = pd.NamedAgg(column='nirDrop', aggfunc=percentile(.10)),
                                    i_nir_drop_25 = pd.NamedAgg(column='nirDrop', aggfunc=percentile(.25)),
                                    i_sr_drop_avg = pd.NamedAgg(column='srDrop', aggfunc='mean'),
                                    i_sr_drop_med = pd.NamedAgg(column='srDrop', aggfunc='median'),
                                    i_sr_drop_min = pd.NamedAgg(column='srDrop', aggfunc='min'),
                                    i_sr_drop_stdv = pd.NamedAgg(column='srDrop', aggfunc='std'),
                                    i_sr_drop_10 = pd.NamedAgg(column='srDrop', aggfunc=percentile(.10)),
                                    i_sr_drop_25 = pd.NamedAgg(column='srDrop', aggfunc=percentile(.25)),
                                    i_red_spike_avg = pd.NamedAgg(column='redSpike', aggfunc='mean'),
                                    i_red_spike_max = pd.NamedAgg(column='redSpike', aggfunc='max'),
                                    i_red_spike_med = pd.NamedAgg(column='redSpike', aggfunc='median'),
                                    i_red_spike_stdv = pd.NamedAgg(column='redSpike', aggfunc='std'),
                                    i_red_spike_90 = pd.NamedAgg(column='redSpike', aggfunc=percentile(.90)),
                                    i_red_spike_75 = pd.NamedAgg(column='redSpike', aggfunc=percentile(.75)),
                                    i_blue_spike_avg = pd.NamedAgg(column='blueSpike', aggfunc='mean'),
                                    i_blue_spike_med = pd.NamedAgg(column='blueSpike', aggfunc='median'),
                                    i_blue_spike_max = pd.NamedAgg(column='blueSpike', aggfunc='max'),
                                    i_blue_spike_stdv = pd.NamedAgg(column='blueSpike', aggfunc='std'),
                                    i_blue_spike_90 = pd.NamedAgg(column='redSpike', aggfunc=percentile(.90)),
                                    i_blue_spike_75 = pd.NamedAgg(column='redSpike', aggfunc=percentile(.75)),
                                    i_blue_avg = pd.NamedAgg(column='blueAvg', aggfunc='mean'),
                                    i_blue_max = pd.NamedAgg(column='blueMax', aggfunc='median'),
                                    i_blue_min = pd.NamedAgg(column='blueMin', aggfunc='median'),
                                    i_blue_stdv = pd.NamedAgg(column='blueStdv', aggfunc='mean'),
                                    i_green_avg = pd.NamedAgg(column='greenAvg', aggfunc='mean'),
                                    i_green_max = pd.NamedAgg(column='greenMax', aggfunc='median'),
                                    i_green_min = pd.NamedAgg(column='greenMin', aggfunc='median'),
                                    i_green_stdv = pd.NamedAgg(column='greenStdv', aggfunc='mean'),
                                    i_red_avg = pd.NamedAgg(column='redAvg', aggfunc='mean'),
                                    i_red_max = pd.NamedAgg(column='redMax', aggfunc='median'),
                                    i_red_min = pd.NamedAgg(column='redMin', aggfunc='median'),
                                    i_red_stdv = pd.NamedAgg(column='redStdv', aggfunc='mean'),
                                    i_nir_avg = pd.NamedAgg(column='nirAvg', aggfunc='mean'),
                                    i_nir_max = pd.NamedAgg(column='nirMax', aggfunc='median'),
                                    i_nir_min = pd.NamedAgg(column='nirMin', aggfunc='median'),
                                    i_nir_stdv = pd.NamedAgg(column='nirStdv', aggfunc='mean'),
                                    i_sr_avg = pd.NamedAgg(column='srAvg', aggfunc='mean'),
                                    i_sr_max = pd.NamedAgg(column='srMax', aggfunc='median'),
                                    i_sr_min = pd.NamedAgg(column='srMin', aggfunc='median'),
                                    i_sr_stdv = pd.NamedAgg(column='srStdv', aggfunc='mean'),
                                    i_ndvi_avg = pd.NamedAgg(column='ndviAvg', aggfunc='mean'),
                                    i_ndvi_max = pd.NamedAgg(column='ndviMax', aggfunc='median'),
                                    i_ndvi_min = pd.NamedAgg(column='ndviMin', aggfunc='median'),
                                    i_ndvi_stdv = pd.NamedAgg(column='ndviStdv', aggfunc='mean'),
                                    i_bsi_avg = pd.NamedAgg(column='bsiAvg', aggfunc='mean'),
                                    i_bsi_max = pd.NamedAgg(column='bsiMax', aggfunc='median'),
                                    i_bsi_min = pd.NamedAgg(column='bsiMin', aggfunc='median'),
                                    i_bsi_stdv = pd.NamedAgg(column='bsiStdv', aggfunc='mean'))
    
    field_df4 = pd.merge(left=field_df0, right=field_df1, left_on='field_id', right_on='field_id')
    field_df = pd.merge(left=field_df4, right=field_df2, left_on='field_id', right_on='field_id') 

    #Add labels
    if label_source != "":
        training_fields = pd.read_csv(label_source)
        training_fields['unique_id'] = training_fields['unique_id'].apply(str)
        field_df_labeled = pd.merge(field_df, training_fields[['unique_id','pv_burnt_any']], left_on='field_id', right_on='unique_id')
        field_df_labeled.reset_index()
        field_df_labeled.rename(columns={'pv_burnt_any': 'label'}, inplace=True)
        field_df_labeled.rename(columns={'unique_id': 'field_id'}, inplace=True)
        pd.DataFrame.to_csv(field_df_labeled, os.path.join(out_dir,'FieldData_labeled.csv'), sep=',', na_rep='NaN', index=True)
        return field_df_labeled
    else:
        pd.DataFrame.to_csv(field_df, os.path.join(out_dir,'FieldData.csv'), sep=',', na_rep='NaN', index=False)
        return field_df

def print_image_list (in_dir, out_dir):
    images = [im for im in os.listdir(in_dir)]
    all_images = pd.DataFrame(images)
    pd.DataFrame.to_csv(all_images, os.path.join(out_dir,'All_images.csv'), sep=',', na_rep='NaN', index=True)

#pixeldf = 
#out_dir = 
#get_coordinate_df (pixeldf, out_dir)

