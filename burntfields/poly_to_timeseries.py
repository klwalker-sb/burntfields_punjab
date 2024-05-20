#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
import sklearn
import csv
import operator
from affine import Affine
from pyproj import Proj, transform
import datetime
import random
import math


def get_ts_stats_for_polys(data_dir, field_list):
    '''
    Gets average and maximum observation gap for time series
    {data_dir} consists of 1 image for each observation with name format fid_YYYMMMDD_anything.tif
    {field_list} is .csv file with column 'unique_id' with fids for fields to include
    '''
    fields = pd.read_csv(field_list)
    avg_delta_d = []
    max_delta_d = []
    for i in range(len(fields)):
        id = fields['unique_id'][i] ## getting the polygon id
        ts=[]
        for j in os.listdir(data_dir):
            img_fid = j[:j.index("_")] #Gets first portion of name prior to first dash
            if str(id) == str(img_fid):
                ##Filter out images from last 2 weeks of year, in case downloaded and not yet removed
                i_date = j[j.index("_")+1:j.index("_")+9]
                imday = int(i_date[-4:])
                if imday < 1216:
                    ts.append(i_date)
        tsd = sorted(set(ts))
        tsd_a = np.array(tsd)
        tsd_t = pd.to_datetime(tsd_a.astype(str)).values
        gap = np.diff(tsd_t).astype('timedelta64[h]')
        if len(gap)>0:
            avg_gap = (np.mean(gap).astype(int))/24
            max_gap = (np.max(gap).astype(int))/24
            avg_delta_d.append(avg_gap)
            max_delta_d.append(max_gap)
    
    print(avg_delta_d)
    
    return avg_delta_d, max_delta_d


def get_pix_coords(ras_file, print_coords = False, out_dir=None):
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
    t0 = ras.transform #this gives coord of top left corner of raster
    t1 = t0 * Affine.translation(0.5, 0.5) #this shifts coord from top corner to center of cell
    p1 = Proj(ras.crs) #to check coordinate system
    #print(p1)
    rc2xy = lambda ras, c: (c, ras) * t1 #gets coordinate for center of cell in x,y position
    #print(rc2xy(0, 0))
    w = ras.width
    h = ras.height
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    eastings, northings = np.vectorize(rc2xy, otypes=[np.float64, np.float64])(rows, cols)
    eastings = eastings.flatten()
    northings = northings.flatten() 
    blue_vals = ma.masked_array(ras.blue, dtype='float64')
    blue_vals = blue_vals.flatten()
    poly_pix = {}
    for i in range(len(blue_vals)):
        if blue_vals[i] is ma.masked:
            pass
        else:
            poly_pix[i] = {}
            poly_pix[i]['Xcoord'] = eastings[i]
            poly_pix[i]['Ycoord'] = northings[i]
            
    #Nested dictionary to dataframe:
    if print_coords == True:
        coord_df = pd.DataFrame.from_dict(poly_pix,orient='index')
        pd.DataFrame.to_csv(coord_df, os.path.join(out_dir,'CoordCheck.csv'), sep=',', na_rep='NaN', index=True)
    
    return poly_pix


def get_date(image_file):
    '''
    Gets date-time object for a file with form idXXXX_YYYYMMDDzzzzz
    where idXXXX is an id of variable length, YYYYMMDD is the image date, and zzzzz are any remaining characters of variable length
    '''
    img = os.path.basename(image_file)
    img_fid = img[:img.index("_")] #Gets first portion of name prior to first dash
    i_date = img[img.index("_")+1:img.index("_")+9]
    imday = i_date[-4:]
    #print(imday)
    return datetime.datetime.strptime(i_date, "%Y%m%d").date()


def poly_data_to_dict (ids, data_dir, num_bands, out_dir, load_bands):
    '''
    'ids' is a csv file with ploygon IDs(with heading 'poly_id')
        and dates (format YYYMMDD) of pre and post-burn observations(with headings 'preburn' and 'burn')
    Images cropped to each polygon are in data_dir and are named PolyID_PlanetID or PolyID_SentinelID(reformatted)
        PolyID is 10 digits for ground-verified polygons but 1-4 digits for newly digitized polygons
        PlanetIDs have names such as: '20191016_052351_0f17_3B' Date(YYYYMMDD is first 8 digits)
        SentineIDs have been rearranged to names such as '20191010T053731_T43RDP' so that Date(YYYMMDD) is also first 8 digits
    Works with 4-band image stacks (blue,green,red,NIR) from Planet, Sentinel 10m, etc.
    or 9-band image stacks (Sentinel 20m). nir=Band8A or 3-band image stacks (BASMA)
    For each polygon, get coordinates and list of images with all values for each band
    Output is nested distionary in the format:
    {PolyID:{Coords:{CoordX, CoordY}}Images:{0:{blue:{values},green:{values},red:{values},nir{values}},1:{..}....}}
    '''
    ## For each polygon in training list:
    fields = pd.read_csv(ids)
    if num_bands == 3:
        band_names = ['blue', 'NPV', 'char']
    if num_bands == 4:
        band_names = ['blue', 'green', 'red', 'nir']
    elif num_bands == 9:
        band_names = ['blue', 'green', 'red', 'redEdge1', 'redEdge2', 'redEdge3', 'nir', 'SWIR1', 'SWIR2']
            
    poly_data = {} #Outer dictionary To contain polyID, image ids and arrays covering all pixels for 4 or 9 band TS
    ##make blank lists to store ids that are missing images
    blank_ids = []

    for i in range(len(fields)):
        id = fields['poly_id'][i] ## getting the polygon id
        print("Working on {} with id {}". format(i, id))
        poly_data[id] = {}
        poly_data[id]['Obs_preburn'] = fields['preburn'][i]
        poly_data[id]['Obs_postburn'] = fields['burn'][i]
        img_count = 0
        good_img_count = 0
        ##For each image in dataset:
        poly_data[id]['images']={}
        num_pix=[]
        for j in os.listdir(data_dir):
            img_fid = j[:j.index("_")] #Gets first portion of name prior to first dash
            if str(id) == str(img_fid):
                ##Filter out images from last 2 weeks of year, in case downloaded and not yet removed
                i_date = j[j.index("_")+1:j.index("_")+9]
                imday = int(i_date[-4:])
                if imday < 1216:
                    input_filename = os.path.join(data_dir,j)
                    try:
                        with rio.open(input_filename) as ras:
                            ras.blue = ras.read(1)
                    except:
                        print("problem processing image {}. skipping.".format(j))
                        img_count = img_count+1
                        pass
                    else:
                        poly_data[id]['images'][img_count] = dict.fromkeys(band_names)
                        poly_data[id]['images'][img_count]['Date']=i_date
                        with rio.open(input_filename) as ras:
                            ras.blue = ras.read(1, masked=True)
                            if 'green' in load_bands:
                                ras.green = ras.read(2, masked=True)
                            if 'red' in load_bands:
                                ras.red = ras.read(3, masked=True)
                            if 'nir' in load_bands:
                                if num_bands == 4:
                                    ras.nir = ras.read(4, masked=True)
                                elif num_bands == 9:
                                    ##(Main NIR is Band 8A in 9-band Sentinel data)
                                    ras.nir = ras.read(7, masked=True)
                            if 'redEdge1' in load_bands:
                                ras.redEdge1 = ras.read(4, masked=True)
                            if 'redEdge2' in load_bands:
                                ras.redEdge2 = ras.read(5, masked=True)
                            if 'redEdge3' in load_bands:
                                ras.redEdge3 = ras.read(6, masked=True)
                            if 'SWIR1' in load_bands:
                                ras.SWIR1 = ras.read(8, masked=True)
                            if 'SWIR2' in load_bands:
                                ras.SWIR2 = ras.read(9, masked=True)
                            if 'BASMA_NPV' in load_bands:
                                ras.NPV = ras.read(2, masked=True)
                            if 'BASMA_char' in load_bands:
                                ras.char = ras.read(3, masked=True)
    
                        ##Get flattened arrays (ordered lists) for each loaded band
                        #if 'blue' in loadBands: #Always load blue as sample raster
                        blue_vals = ma.masked_array(ras.blue, dtype='float64')
                        ras_pix = ma.masked_array.count(blue_vals)
                        num_pix.append(tuple([input_filename, ras_pix]))
                        blue_vals = blue_vals.flatten()
                        blue_vals = blue_vals.filled(np.NAN) 
                        poly_data[id]['images'][img_count]['blue']=blue_vals
                        #PolyPix = np.count_nonzero(~np.isnan(blueVals)) #Already counting the masked array above
                        #print("{} frame pix and {} poly pix in img {}".format(RasPix, PolyPix, imgCount))
                        if 'green' in load_bands:
                            green_vals = ma.masked_array(ras.green, dtype='float64')
                            green_vals = green_vals.flatten()
                            green_vals = green_vals.filled(np.NAN) 
                            poly_data[id]['images'][img_count]['green']=green_vals
                        if 'red' in load_bands:
                            red_vals = ma.masked_array(ras.red, dtype='float64')
                            red_vals = red_vals.flatten()
                            red_vals = red_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['red']=red_vals
                        if 'nir' in load_bands:
                            nir_vals = ma.masked_array(ras.nir, dtype='float64')
                            nir_vals = nir_vals.flatten()
                            nir_vals = nir_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['nir']=nir_vals
                        if 'redEdge1' in load_bands:
                            red_edge1_vals = ma.masked_array(ras.redEdge1, dtype='float64')
                            red_edge1_vals = red_edge1_vals.flatten()
                            red_edge1_vals = red_edge1_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['redEdge1']=red_edge1_vals
                        if 'redEdge2' in load_bands:
                            red_edge2_vals = ma.masked_array(ras.redEdge2, dtype='float64')
                            red_edge2_vals = red_edge2_vals.flatten()
                            red_edge2_vals = red_edge2_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['redEdge2']=redE_edge2_vals
                        if 'redEdge3' in load_bands:
                            red_edge3_vals = ma.masked_array(ras.redEdge3, dtype='float64')
                            red_edge3_vals = red_edge3_vals.flatten()
                            red_edge3_vals = red_edge3_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['redEdge3']=red_edge3_vals
                        if 'SWIR1' in load_bands:
                            swir1_vals = ma.masked_array(ras.SWIR1, dtype='float64')
                            swir1_vals = swir1_vals.flatten()
                            swir1_vals = swir1_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['SWIR1']=swir1_vals
                        if 'SWIR2' in load_bands:
                            swir2_vals = ma.masked_array(ras.SWIR2, dtype='float64')
                            swir2_vals = swir2_vals.flatten()
                            swir2_vals = swir2_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['SWIR2']=swir2_vals
                        if 'BASMA_NPV' in load_bands:
                            npv_vals = ma.masked_array(ras.NPV, dtype='float64')
                            npv_vals = npv_vals.flatten()
                            npv_vals = npv_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['NPV']=npv_vals
                        if 'BASMA_char' in load_bands:
                            char_vals = ma.masked_array(ras.char, dtype='float64')
                            char_vals = char_vals.flatten()
                            char_vals = char_vals.filled(np.NAN)
                            poly_data[id]['images'][img_count]['char']=char_vals
                            
                        img_count = img_count+1
                        good_img_count = good_img_count + 1
        if good_img_count == 0:
            blank_ids.append(id)
            print("there are no valid images for field {}".format(id))
        elif good_img_count > 0:
            ##Get coordinates from image with max # of unmasked pixels (because clouds were masked prior to clipping imagery)
            max_pix = max(num_pix, key=operator.itemgetter(1)) 
            poly_data[id]['coords'] = getPixCoords(max_pix[0])
            
    if len(blank_ids)>0:
        blank_file = open(os.path.join(out_dir,'ids_without_images.txt'),'w')
        for element in blank_ids:
            blank_file.write(str(element) + "\n")
        print("there are {} ids with no images, written to file:{}".format(len(blank_ids),blank_file))

    return poly_data

def sample_pixels_from_array (array, samp_list, pix_ids):
    matrix1 = pd.DataFrame(array)
    matrix1 = matrix1.dropna(axis='columns', how='all')
    matrix2 = matrix1[matrix1.columns.intersection(samp_list)]
    matrix2 = matrix2.set_axis(pix_ids, axis=1)
    return matrix2
    
def matrix_to_timeSeries(poly_id, matrix2, date_list, preburn, postburn):
    '''
    Converts Matrix to DataFrame with row index = image date and column index = polygon ID
    One entry per day (even if images not availabe for given day) to create time series and enable resampling to any frequency
    by doing so, also deals with images from duplicate dates by averaging values (or filling missing values)
    '''
    matrix2.index=date_list
    matrix2.index = pd.to_datetime(matrix2.index)
    daily_ts = matrix2.resample('D').mean()
    ##Add "ALL" column, which is the average value for all polygons (average of averages)
    avg_name = 'AVG_'+ str(poly_id)
    daily_ts[avg_name] = daily_ts.mean(axis=1)

    ##Calculate length of burn observation window(from last preburn obs to first post-burn obs)
    burnwindow = postburn-preburn
    #print('burnwindow = {}'.format(burnwindow))
    ##Assign burndate to midpoint (a little arbitrary) and first day after burnobs 
    est_days_from_burn0 = int(math.ceil(burnwindow.days/2)) #rounds up, so 1 if 1 day window
    postburn = pd.to_datetime(postburn)
    preburn = pd.to_datetime(preburn)
    
    ##Create new columns to reindex time series based on days from burn observation
    daily_ts['BurnTS']=0
    daily_ts['FromBurn']=(daily_ts.index.dayofyear-postburn.dayofyear)+est_days_from_burn0
    daily_ts['PreBurn']=(daily_ts.index.dayofyear-preburn.dayofyear)-est_days_from_burn0
    daily_ts.loc[daily_ts.index >= postburn, 'BurnTS'] = daily_ts.FromBurn
    daily_ts.loc[daily_ts.index <= preburn, 'BurnTS'] = daily_ts.PreBurn
    daily_ts.drop(['FromBurn','PreBurn'], axis=1, inplace=True)
    ##extend range from -50 to 50:
    max_obs = daily_ts['BurnTS'].max()
    min_obs = daily_ts['BurnTS'].min()
    for i in range(max_obs+1,51):
        daily_ts = daily_ts.append({'BurnTS' : i},ignore_index=True)
    for i in range(-50, min_obs):
        daily_ts = daily_ts.append({'BurnTS' : i},ignore_index=True)   
    daily_ts.set_index('BurnTS', inplace=True)
    daily_ts.sort_index(inplace=True)
    ##For testing:
    #pd.DataFrame.to_csv(daily_ts, os.path.join(out_dir,'TimeSeries_test.csv'), sep=',', na_rep='NaN', index=True)

    #break dataframe into nested dictionary ({pixel:{TS_val:data_obs}}) to add to giant list and convert to df in end
    ts_dict = daily_ts.to_dict()
    return ts_dict


def pixel_level_calcs(poly_id, poly_dict, num_bands, out_dir, num_samp, measure, load_bands):
    '''
    Gets a Time Series of a given measure for a single polygon...
    '''
    
    #poly_data[poly_id] has been passed as poly_dict
    preburn_obs = datetime.datetime.strptime(str(poly_dict['Obs_preburn']), "%Y%m%d").date()
    postburn_obs = datetime.datetime.strptime(str(poly_dict['Obs_postburn']), "%Y%m%d").date()
    print(postburn_obs)
    date_list = []
    blue_arrays=[]
    if num_bands == 3:
        npv_arrays=[]
        char_arrays=[]
    if num_bands >3:
        green_arrays=[]
        red_arrays=[]
        nir_arrays=[]
    if num_bands == 9:
        rededge1_arrays=[]
        rededge2_arrays=[]
        rededge3_arrays=[]
        swir1_arrays=[]
        swir2_arrays=[]
        
    ##For each image, get pixel values for loaded bands and append to band array holding pixel-values for all images:   
    for key, value in poly_dict['images'].items():
        #print("second dict key (image): {}".format(key))
        img_id = key
        if poly_dict['images'][img_id] == None:
            pass
        else:
            img_date = datetime.datetime.strptime(poly_dict['images'][img_id]['Date'], "%Y%m%d").date()
            date_list.append(img_date)
            blue_series = poly_dict['images'][img_id]['blue']
            blue_arrays.append(blue_series)
            if 'green' in load_bands: 
                green_series = poly_dict['images'][img_id]['green']
                green_arrays.append(green_series)
            if 'red' in load_bands: 
                red_series = poly_dict['images'][img_id]['red']
                red_arrays.append(red_series)
            if 'nir' in load_bands:
                nir_series = poly_dict['images'][img_id]['nir']
                nir_arrays.append(nir_series)
            if 'redEdge1' in load_bands:
                rededge1_series=poly_dict['images'][img_id]['redEdge1']
                rededge1_arrays.append(rededge1_series)
            if 'redEdge2' in load_bands:
                rededge2_series=poly_dict['images'][img_id]['redEdge2']
                rededge2_arrays.append(rededge2_series)
            if 'redEdge3' in load_bands:
                rededge3_series=poly_dict['images'][img_id]['redEdge3']
                rededge3_arrays.append(rededge3_series)
            if 'SWIR1' in load_bands:
                swir1_series=poly_dict['images'][img_id]['SWIR1']
                swir1_arrays.append(swir1_series)
            if 'SWIR2' in load_bands:
                swir2_series=poly_dict['images'][img_id]['SWIR2']
                swir2_arrays.append(swir2_series)
            if 'BASMA_char' in load_bands:
                char_series=poly_dict['images'][img_id]['char']
                char_arrays.append(char_series)
            if 'BASMA_NPV' in load_bands:
                npv_series=poly_dict['images'][img_id]['NPV']
                npv_arrays.append(npv_series)
        
    ##Convert final array to matrix with one row for each image and one column for each pixel and get summary calcs       
    blue_matrix = pd.DataFrame(blue_arrays)
    blue_matrix = blue_matrix.dropna(axis='columns', how='all')
    
    ##Don't want all the pixels here. Sample N random pixels
    pixels = blue_matrix.columns.tolist()
    if len(pixels)==0:
        pass
    else:
        #print("This polygon has {} pixels. Sampling {} of them".format(len(pixels), numSamp))
        keep_pix = []
        pix_ids = []
        drop_pixels = []
        ##Sample desired number of pixels at once, so duplication is not possible (duplicates cause error later)
        if len(pixels) < num_samp:
            num_samp = len(pixels)
        else:
            num_samp = num_samp
        rani = np.random.choice(len(pixels),num_samp,replace=False) 
        for i in rani:
            pix_num = pixels[i]
            if pix_num not in poly_dict['coords']:
                drop_pixels.append(pix_num)
                pass
            else:
                pixel_data = {}
                pix_id = str(poly_id)+"_"+str(pix_num)
                #pixel_data[pix_id] = {}
                keep_pix.append(pix_num)
                pix_ids.append(pix_id)
                num_images = np.count_nonzero(~np.isnan(blue_matrix[pix_num]))
                #pixel_data[pix_id]["img_count"]=num_images
                #if num_images <= 13:
                #drop_pixels.append(pix_id)
        #print(pix_ids)
       
        scale = 10000
        
        if 'blue' in load_bands: 
            blue_matrix = sample_pixels_from_array(blue_arrays, keep_pix, pix_ids)
        if 'green' in load_bands: 
            green_matrix = sample_pixels_from_array(green_arrays, keep_pix, pix_ids)
        if 'red' in load_bands: 
            red_matrix = sample_pixels_from_array(red_arrays, keep_pix, pix_ids)
        if 'nir' in load_bands: 
            nir_matrix = sample_pixels_from_array(nir_arrays, keep_pix, pix_ids)
        if 'redEdge1' in load_bands: 
            rededge1_matrix = sample_pixels_from_array(rededge1_arrays, keep_pix, pix_ids)
        if 'redEdge2' in load_bands: 
            rededge2_matrix = sample_pixels_from_array(rededge2_arrays, keep_pix, pix_ids)
        if 'redEdge3' in load_bands: 
            rededge3_matrix = sample_pixels_from_array(rededge3_arrays, keep_pix, pix_ids)
        if 'SWIR1' in load_bands: 
            swir1_matrix = sample_pixels_from_array(SWIR1_arrays, keep_pix, pix_ids)
        if 'SWIR2' in load_bands: 
            swir2_matrix = sample_pixels_from_array(SWIR2_arrays, keep_pix, pix_ids)
        if 'BASMA_NPV' in load_bands: 
            npv_matrix = sample_pixels_from_array(NPV_arrays, keep_pix, pix_ids)
        if 'BASMA_char' in load_bands: 
            char_matrix = sample_pixels_from_array(char_arrays, keep_pix, pix_ids)
            

        ##If raw band measures desired:
        if measure == 'blue':
            mmatrix = blue_matrix
        elif measure == 'green':
            mmatrix = green_matrix
        elif measure == 'red':
            mmatrix = red_matrix
        elif measure == 'nir':
            mmatrix = nir_matrix
        ##Sentinel-only bands:
        elif measure == 'redEdge1':
            mmatrix = rededge1_matrix
        elif measure == 'redEdge2':
            mmatrix = rededge2_matrix
        elif measure == 'redEdge3':
            mmatrix = rededge3_matrix
        elif measure == 'SWIR1':
            mmatrix = swir1_matrix
        elif measure == 'SWIR2':
            mmatrix = swir2_matrix
        ##BASMA bands (green veg is being calculated as blue_matrix)
        elif measure == 'BASMA_char':
            mmatrix = char_matrix
        elif measure == 'BASMA_NPV':
            mmatrix = npv_matrix
        
        ##Else, calculate indices
        
        ##Planet & Sentinel indices (ndvi, bsi(BareSoil), sr, CI)
        elif measure == 'ndvi':
            mmatrix = scale * (nir_matrix - red_matrix)/(nir_matrix + red_matrix)
        elif measure == 'BareSoil':
            mmatrix = scale * (((nir_matrix/scale + green_matrix/scale) - (red_matrix/scale + blue_matrix/scale))/(nir_matrix/scale + green_matrix/scale + red_matrix/scale + blue_matrix/scale))*100+100
        elif measure == 'sr':
            mmatrix = scale * nir_matrix / red_matrix
        elif measure == 'CI':
            vissum = (blue_matrix/scale + green_matrix/scale + red_matrix/scale)
            bg = abs(blue_matrix/scale - green_matrix/scale)
            br = abs(blue_matrix/scale - red_matrix/scale)
            rg = abs(red_matrix/scale - green_matrix/scale)
            maxdiff1 = np.maximum(bg, br)
            maxdiff = np.maximum(maxdiff1, rg)
            mmatrix = scale * (vissum + (maxdiff*15))
        elif measure == 'BAI':
            mmatrix = scale/100 * 1/((.06-(nir_matrix/scale))**2+(.1-(red_matrix/scale))**2)
    
        ##Sentinel-ony indices            
        elif measure == 'NBR':
            mmatrix = scale * (nir_matrix/scale - swir2_matrix/scale)/(nir_matrix/scale + swir2_matrix/scale)
        elif measure == 'NBR2':
            mmatrix = scale * (swir1_matrix - swir2_matrix)/(swir1_matrix + swir2_matrix)
        elif measure == 'MIRBI': #Note, original eq is for 0-1 range. Need to divide band data by 10000 to convert
            mmatrix = scale * (10*(swir2_matrix/scale) - 9.8*(swir1_matrix/scale) + 2)
        elif measure == 'BurnScar': #Note, original eq is for 0-1 range. Need to divide band data by 10000 to convert
            m = 2 #(try 1,2,4,6) until seperability is maximized
            burnbare_ratio = (swir2_matrix - red_matrix)/(swir2_matrix + red_matrix)
            background = ((green_matrix)**m + (red_matrix)**m + (nir_matrix)**m)
            mmatrix = (burnbare_ratio * 1/background)
        elif measure == 'BAIS':
            mmatrix = scale/100 * (1 - (((rededge2_matrix/scale * rededge3_matrix/scale * nir_matrix/scale) / red_matrix/scale))**1/2) * (((swir2_matrix/scale - nir_matrix/scale) /(swir2_matrix/scale + nir_matrix/scale)**1/2) + 1)
        ###Create time series per pixel(column), with burn window (last pre-burn obs to first post-burn obs) set to 0
        pixel_data = matrix_to_timeSeries(poly_id, mmatrix, date_list, preburn_obs, postburn_obs)
    
        return pixel_data


def get_ts_for_all_pixels(field_list, data_dir, num_bands, out_dir, num_samp, measure):
    '''
    Gets a Time Series of a given measure for a set of polygons (field_list) 
    and a directory containing ordered images for each polygon (data_dir). 
    Can set the number of random points sampled from each polygon with numSamp.
    Prints 2 dataframes: 'PixelData' with all sampled pixels and 'PolyData' with average values for each polygon
    Imagery is expected to be in the form of either 4-band stacks (blue, green, red, NIR)
    or 9-band stacks (blue, green, red, NIR_B5, NIR_B6, NIR_B7, NIR_B8A, NIR_B5, SWIR_B11, SWIR_B12)
    specified with NumBands parameter
    measure can be either individual band (as named above), or index.
    '''
    indices = ['red','blue','green','nir','redEdge1', 'redEdge2', 'redEdge3', 'SWIR1', 'SWIR2',
                'ndvi','sr','BareSoil','CI','NBR','NBR2','MIRBI','BurnScar','BAI','BAIS','BASMA_char','BASMA_NPV']
       
    if measure not in indices:
        print('{} is not a current index choice. Choices are: {}'.format(measure, indices))
    
    load_bands = []
    if measure in ['red','blue','green','nir','redEdge1', 'redEdge2', 'redEdge3', 'SWIR1', 'SWIR2','BASMA_char','BASMA_NPV']:
        load_bands = [measure]
    elif measure in ['ndvi','sr','BAI']:
        load_bands = ['red','nir']
    elif measure == 'BareSoil':
        load_bands = ['red','green','blue','nir']
    elif measure == 'CI':
        load_bands = ['red','green','blue']
    elif measure == 'NBR':
        load_bands = ['nir','SWIR2']
    elif measure in ['NBR2','MIRBI']:
        load_bands = ['SWIR1', 'SWIR2']
    elif measure == 'BurnScar':
        load_bands = ['green','red','nir','SWIR2']
    elif measure == 'BAIS':
        load_bands = ['red','redEdge2','redEdge3','nir','SWIR2']

    pixel_sets = {}
    pixel_data = {}
    poly_data = poly_data_to_dict(field_list, data_dir, num_bands, out_dir, load_bands)
    for key, value in poly_data.items():
        print("first dict key (polygon): {}".format(key))
        poly_id = key
        pixel_data[poly_id] = pixel_level_calcs(poly_id, poly_data[poly_id], num_bands, out_dir, num_samp, measure, load_bands)
        if pixel_data[poly_id] == None:
            pass
        else:
            for key, value in pixel_data[poly_id].items():
                print("getting pixel {}".format(key))
                pixel_sets[key]=value
                #print(pixel_sets[band_id])
                
    ##Data sheet with all pixels in sample
    pixel_sheet = pd.DataFrame.from_dict(pixel_sets,orient='columns')
    ##Data sheet with only average values for each polygon
    avg_cols = [col for col in pixel_sheet.columns if 'AVG' in col]
    poly_sheet = pixel_sheet[pixel_sheet.columns.intersection(avg_cols)]
    #pd.DataFrame.to_csv(pixel_sheet, os.path.join(out_dir,'TS_pixelData.csv'), sep=',', na_rep='NaN', index=True)
    pd.DataFrame.to_csv(poly_sheet, os.path.join(out_dir,'TS_polyData_Sentinel_'+measure+'.csv'), sep=',', na_rep='NaN', index=True)
    print("done")
    return poly_sheet


def get_burnNoburn_classes_unmatched(burn_ts, num_days, inclusive=True):
    '''
    Creates two datasets (Burn and PreBurn) for all observations based on time series and desired number of days from burn observation
    NumDays can be exact (e.g. only observations 3 days after burn are considered, 
    or inclusive (observations up to 3 days from burn even are considered) based on inclusive variable)
    '''
    obs_data = pd.read_csv(burn_ts, index_col='FromBurn')

    #Get list of valid observation for each number of days since burn
    obs_data_T = obs_data.transpose()
    from_burn_dict = {}
    for column in obs_data_T:
        val_list = obs_data_T[column].dropna().tolist()
        from_burn_dict[column] = val_list

    burn_set = []
    no_burn_set = []
    if inclusive == True:    
        for i in range(1, num_days+1):
            burn_set.extend(from_burn_dict[i])
            no_burn_set.extend(from_burn_dict[-i])
    else:
        burn_set = from_burn_dict[num_days]
        no_burn_set = from_burn_dict[-num_days]

    print(burn_set)
    print(no_burn_set)
    return burn_set, no_burn_set


def get_burnNoburn_classes_matched(burn_ts, out_dir):
    '''
    TODO: Check dataframe. NA removal method seems wrong (removes all values that don't apply to poly1).
    Can also just use Get_TSforDaysSinceBurn with SinglePostObs=True for cases where time interval is not needed after filtering
    
    Converts time series into matched pre and post-burn observations for each id 
    (polygon or pixel dataframes can be used)
    pre-burn observation is last obs before burn date
    post-burn is first obs on or following burn date
    (burn date is provided in polygon id file that is time-series input 
         and is based on visual inspection of Planet imagery)
    Output dataframe includes 'DaysSinceBurn'(for postburn) and 'ObsInterval'(post-pre) for post-filtering
    '''
    obs_data = burn_ts
    #obsData = pd.read_csv(BurnTS, index_col=0)
    matched_burn = {}
    ##Get first + and first - value for each column (polygon)
    for id in obs_data:
        pre_burn = obs_data[(obs_data.index < 0) & (~obs_data[id].isin([np.nan, ' ']))]
        post_burn = obs_data[(obs_data.index >= 0) & (~obs_data[id].isin([np.nan, ' ']))]
        print(post_burn) #NOTE, this doesn't seem to be working as intended
        if len(pre_burn) == 0 or len(post_burn) == 0:
            pass
        else:
            last_pre_burn = pre_burn.index.max()
            last_pre_burn_val = obs_data.loc[last_pre_burn][id]
            first_post_burn = post_burn.index.min()
            first_post_burn_val = obs_data.loc[first_post_burn][id]
            #post_burn2 = post_burn.index.nsmallest(2)
            #post_burn2_val = obsData.loc[post_burn2][id]
            burn_interval = first_post_burn - last_pre_burn
            matched_burn[id]={}
            matched_burn[id]['PreBurn']= last_pre_burn_val
            matched_burn[id]['PostBurn']= first_post_burn_val
            matched_burn[id]['DaysSinceBurn'] = first_post_burn
            #matched_burn[id]['PostBurn2']= PostBurnVal2
            #matched_burn[id]['DaysSinceBurn2'] = PostBurn2
            matched_burn[id]['ObsInterval'] = burn_interval
    matched_burn_df = pd.DataFrame.from_dict(matched_burn,orient='index')
    pd.DataFrame.to_csv(matched_burn_df, os.path.join(out_dir,'MatchedBurnObs.csv'), sep=',', index=True)
    
    return matched_burn_df


def get_ts_for_days_since_burn(burn_ts, out_dir, num_days_pre, num_days_post1, num_days_post2, single_post_obs, print_df=True):
    '''
    Converts full TS dataframes into days since burn for 0-<num_days_post>days post burn.
    Gets pre-burn obs from first <num_days_pre> days before burn to compare.
    Output dataframe is columns -1 (pre_burn) and each day since up to <num_days_post> 
    with obs value for each (where available) for each field (row)
    If <single_post_obs> == True, post observation is single cloumn with first value since event
    '''
    obs_data = burn_ts
    #obsData = pd.read_csv(burn_ts, index_col=0)
    
    ## Narrow time series <num_days_pre> days preburn and <numDaysPost2> days post burn:
    obs_data_narrow = obs_data[(obs_data.index >= -1*num_days_pre) & (obs_data.index <num_days_post2)]
    ts = obs_data_narrow.transpose()
    
    ## Get preburn value (if within <numDaysPre> days of burn obs)
    pre_burn = ts[range(-1*num_days_pre,0,1)]
    post_burn = ts[range(num_days_post1,num_days_post2,1)]
    ## Fill NAs such that most recent preburn obs is in -1 position
    pre_burn_f = pre_burn.ffill(axis=1)
    pre_burn_val = pre_burn_f[[-1]]
    
    if single_post_obs == True:
        post_burn_f = post_burn.bfill(axis=1)
        post_burn_val = post_burn_f[[num_days_post1]]
        tsdsb = pd.concat([pre_burn_val, post_burn_val], axis=1)
        tsdsb.rename({-1:'Pre', num_days_post1:'Post'}, axis=1, inplace=True)
        if print_df == True:
            pd.DataFrame.to_csv(tsdsb, os.path.join(out_dir,'PrePost.csv'), sep=',', index=True)
    else:
        tsdsb = pd.concat([pre_burn_val, post_burn], axis=1)
        if print_df == True:
            pd.DataFrame.to_csv(tsdsb, os.path.join(out_dir,'DaysSinceBurn.csv'), sep=',', index=True)
    
    return tsdsb

def get_image_diffs(ts):
    '''
    takes a time series and returns difference matrix for V(t+1)-V(t) where t is the 
    '''
    ts_f = ts.ffill(axis=0)
    ts_t2 = ts_f.copy()
    ts_t2['up_one']=ts_t2.index - 1
    ts_t2.set_index(ts_t2['up_one'], inplace=True)
    diff = ts_t2 - ts_f
    diff.drop(columns=['up_one'],inplace=True)
    
    return diff

def get_false_signals(diff_ts, burn_sig_dict):
    '''
    compares the difference time series (V(t+1)-V(t)) to the burn signal (V(pre-burn) - V(post-burn) for N days post burn)
    returns a list of the percentage of pixels or polygons with noise > signal at at least one point in the seires
    '''
    false_sig = []
    for key, value in burn_sig_dict.items():
        sig_noise = diff_ts.gt(value).sum()
        false_sig.append(sig_noise.gt(0).sum() / len(sig_noise))
    return false_sig

def get_burn_signal(ts_dir, sensor, var, max_days_pre, max_days_post):
    '''
    gets avg pre-post value by day since burn, for up to {max_days_post} for {var} and {sensor}
    {max_days_pre} defines the number of days that can be searched pre burn event before the observation is excluded.
    assumes that ts_dataframe has already been created with GetTSforDaysSinceBurn and printed into {ts_dir}
    '''
    burn_ts = pd.read_csv(os.path.join(ts_dir,'TS_polyData_{}_{}.csv'.format(sensor,var)),index_col=0)
    sig_dic = {}
    for n in range(2,max_days_post):
        tsdsb = get_ts_for_days_since_burn(burn_ts, None, max_days_pre, n-1, n, True, print_df=False)
        tsdsb['diff']=tsdsb['Pre']-tsdsb['Post']
        avg_dif = tsdsb["diff"].mean()
        sig_dic[n] = avg_dif
    return sig_dic

def get_noise_vs_signal(ts_dir, sensor, var, max_days_pre, max_days_post):
    burn_ts = pd.read_csv(os.path.join(ts_dir,'TS_polyData_{}_{}.csv'.format(sensor,var)),index_col=0)
    sig_dict = get_burn_signal(ts_dir, sensor, var, max_days_pre, max_days_post)

    pre_burn = burn_ts[(burn_ts.index < 0)]
    pre_burn_diff = get_image_diffs(pre_burn)
    print(pre_burn_diff)
    false_sig_pre = get_false_signals(pre_burn_diff, sig_dict)

    post_burn = burn_ts[(burn_ts.index > 0)]
    post_burn_diff = get_image_diffs(post_burn)
    false_sig_post = get_false_signals(post_burn_diff, sig_dict)

    return false_sig_pre, false_sig_post

def separability_measures(matched_burn_df):
    '''
    see https://blog.actorsfit.com/a?ID=00600-d8fccf3c-8744-41a2-b2c5-b302354affb6
    '''
    matched_burn_df.reset_index(drop=True, inplace=True) 
    p = matched_burn_df['PreBurn']
    q = matched_burn_df['PostBurn']
    
    ##parametric separability index (M) (Kaufman and Remer, 1994, good if >1)
    sep = abs(np.mean(p) - np.mean(q)) / (np.std(p)+np.std(q))
    print('separability (M) = {}'.format(sep))
    
    ##Bhattacharyya coefficient
    BC=np.sum(np.sqrt(p*q))
    ##Bhattacharyya distance
    b=-np.log(BC)
    print('Bhattacharyya distance = {}'.format(b))
    
    ##Jeffries-Matusita distance
    jm= 2*(1 - math.exp(-b)) 
    print('Jeffries-Matusita distance = {}'.format(jm))
    
    ##Jensen-Shannon divergence
    M=(p+q)/2
    js=0.5*np.sum(p*np.log(p/M))+0.5*np.sum(q*np.log(q/M))
    print('Jensen-Shannon divergence = {}'.format(js))

    ##f divergence
    def f(t):
        return t*np.log(t)
    f1=np.sum(q*f(p/q))
    print('f divergence = {}'.format(f1))


def separability_ts(tsbdf):
    '''
    see https://blog.actorsfit.com/a?ID=00600-d8fccf3c-8744-41a2-b2c5-b302354affb6
    '''
    tsbdf.reset_index(drop=True, inplace=True) 
    p = tsbdf[-1.0]
    M = []
    for nd in range(1,15):
        q = tsbdf[nd]
        
        ##parametric separability index (M) (Kaufman and Remer, 1994, good if >1)
        sep = abs(np.mean(p) - np.mean(q)) / (np.std(p)+np.std(q))
        M.append(sep)
       
    mdf = pd.DataFrame(M)
    mdf.index += 1 
    print(mdf)
    return mdf
