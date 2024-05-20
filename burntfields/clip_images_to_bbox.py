#!/usr/bin/env python
# coding: utf-8
# %%
import os
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import geojson
import json
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
import rasterstats
import fiona
from fiona.crs import from_epsg
from xml.dom import minidom
import shutil

#For large datasets / parallel processing:
import time
import multiprocessing
#from pyproj import Proj
#import pycrs

###################################################################################################################################
## Main process: clip_images_to_AOIs(AOI_file, in_dir, out_dir)
##      Specifically for Planet data (requires .xml file with tags such as 'ps:bottomLeft'{'longitude','latitude' and 'ps:topRight')
##      creates temporal stack of imagery for each polygon by clipping each image to each polygon
##      (clips first to bounding box around polygon, and later in process (in another script) to polygons themselves.
##       this is inefficient and only being done to adhere to original code structure. 
## 
##
##INPUTS
## **in_dir** is path to folder with all images to be processed
## **out_dir** is path to folder to store clipped images
## **aoi_file** is .csv file with polygons to process with the following fields:
##        "unique_id" "minLat" "maxLat" "minLon" "maxLon"
##        these are coordinates for 600m bounding boxes, pre-derived in ArcGIS(or similar)
##        (note: 600m bounding boxes are what Jenny used in the original code. There is no practical reason for 600m
##         in fact, 600m sometimes cuts fields (I expanded these manually for V2). In general, I suggest getting 
##        coordinates of fitted bounding box enevelopes in ArcGIS(or similar) rather than using an arbitrary distance)
## --or--
## **field_shapefile** For alternative direct clipping method
## 


def list_files(in_dir, endstring):
# Relevant endstrings: "AnalyticMS.tif", "toar.tif", "udm.tif", "udm2.tif", "metadata.xml"
    files = [os.path.join(in_dir,f) for f in os.listdir(in_dir) if f.endswith(endstring)]
    return files

### Check projections:

def match_projections(shapefile, raster_dir):
    '''
    Checks if shapefile and raster coordinate systems match
    If coordinate systems do not match, reprojects shapefile to match rasters
    saves new shapefile in out directory and returns path
    '''       
    shapes = gpd.read_file(shapefile)
    print ("original shapefile crs is: {}".format(shapes.crs)) #epsg: 4326 (this is lat-lon)
    
    test_img =os.path.join(raster_dir, os.listdir(raster_dir)[1])
    with rio.open(test_img) as samp_img:
        img_crs = samp_img.crs
        print("Images are in:", img_crs) #espg 32643 (this is UTM 43N)

    if shapes.crs != img_crs:
        # re-project shapefile to match images
        dst_crs = img_crs
        shapes_transform = shapes.to_crs(dst_crs) #note, this is a geodataframe
        print ("new shapefile crs is: {}".format(shapes_transform.crs))
        #save reprojected file as shapefile
        new_shapes = shapes_transform.to_file(os.path.join(os.path.dirname(shapefile), 'shape2_T'))
        field_poly_path = os.path.join(os.path.dirname(shapefile), 'shape2_T')
    else: field_poly_path = shapefile
        
    return field_poly_path


def clip_images_to_AOIs(aoi_file, in_dir, out_dir):
    '''
    Compares polygon bounding box coordinates (in AOI_file) to coordinates in metadata to check for overlap.
    AOI file must contain fields with headings: 'unique_id', 'minLat', 'minLon', 'maxLat', 'maxLon'
    If overlap occurs, clips image and corresponding cloudmasks to bounding box. Attaches unique field ID to 
    file names and outputs them, along with associated metadata, to out folder.
    This requires that .xml metadata file with same names accompany .tif files in the same folder.
    '''
    #Check coordinate system of images:
    with rio.open(list_files(in_dir, ".tif")[1]) as samp_img:
        img_crs = samp_img.crs
        #print("Images are in:", img_crs)
  
    locations = pd.read_csv(aoi_file)
    loc_id = locations['unique_id']
    poly_minlon = locations['minLon'].values.tolist()
    poly_minlat = locations['minLat'].values.tolist()
    poly_maxlon = locations['maxLon'].values.tolist()
    poly_maxlat = locations['maxLat'].values.tolist()
    xmlfiles = []
    for id in range(len(loc_id)):
        print('processing polygon #{} with id {}, starting at:{}'.format(id, loc_id[id], datetime.now()))
        #print ("For poly: minlon = {}, minlat = {}, maxlon = {}, maxlat = {}".format(poly_minlon[id], poly_minlat[id], poly_maxlon[id], poly_maxlat[id]))
        poly_bbox = box(poly_minlon[id], poly_minlat[id], poly_maxlon[id],poly_maxlat[id])
        #Set up geography for clipping functions
        geo = gpd.GeoDataFrame({'geometry': poly_bbox}, index=[0], crs=from_epsg(4326))
        #Need to reproject to coordinate system of imagery
        geo = geo.to_crs(crs=img_crs)
        geom = [json.loads(geo.to_json())['features'][0]['geometry']]
        fid = str(loc_id[id])        
        
        for index, fp in enumerate(list_files(in_dir,".xml")):
            base_id = str(os.path.basename(fp)[:21]) #The first 21 digits are always unique
            file = os.path.join(in_dir,fp)
            xmlfiles.append(file)
            #Read metadata to check for overlap in footprint:
            if fp.endswith('.xml'):
                try:
                    metaparse = minidom.parse(fp)
                except:
                    print ('error reading metadata for {}'.format(fp))
                    pass
                else:
                    img_minlon = float(metaparse.getElementsByTagName(
                        "ps:bottomLeft")[0].getElementsByTagName("ps:longitude")[0].firstChild.data)
                    img_minlat = float(metaparse.getElementsByTagName(
                        "ps:bottomLeft")[0].getElementsByTagName("ps:latitude")[0].firstChild.data)
                    img_maxlon = float(metaparse.getElementsByTagName(
                        "ps:topRight")[0].getElementsByTagName("ps:longitude")[0].firstChild.data)
                    img_maxlat = float(metaparse.getElementsByTagName(
                        "ps:topRight")[0].getElementsByTagName("ps:latitude")[0].firstChild.data)
                    #print ("For image: minlon = {}, minlat = {}, maxlon = {}, maxlat = {}".format(img_minlon, img_minlat, img_maxlon, img_maxlat))
            
            #check if raster overlaps polygon (ignore if it doesn't):
                if (img_maxlat < poly_minlat[id] 
                    or img_maxlon < poly_minlon[id] 
                    or img_minlon > poly_maxlon[id] 
                    or img_minlat > poly_maxlat[id]):
                    #Intersection = Empty
                    pass
                else:
                    print ("found an intersection with image {}".format(base_id))
                    #Intersection = NotEmpty
                    #If there is overlap, get corresponding .tif files and crop them to poly bbox
                    for tif in list_files(in_dir, ".tif"):
                        if str(os.path.basename(tif)[:21]) == base_id: 
                            clip_name = fid + "_" + str(os.path.basename(tif))
                            out_img_name = os.path.join(out_dir, clip_name)
                            print(out_img_name)
                            if os.path.isfile(out_img_name):
                                print('file already exists')
                            else:
                                try:
                                    with rio.open(tif) as src:
                                        a_band = src.read(1)
                                except:
                                    print ("WARNING: Could not open file {}".format(tif))
                                    pass
                                else:
                                    try:
                                        with rio.open(tif) as src:
                                            _band = src.read(1)
                                            out_image, out_transform = mask(src, geom, crop=True)
                                            out_meta = src.meta.copy()
                                            #epsg_code = int(src.crs['init'][5:])
                                            out_meta.update({"driver": "GTiff",
                                                "height": out_image.shape[1],
                                                "width": out_image.shape[2],
                                                "transform": out_transform,
                                                "crs": img_crs})
                                                #"crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()})
                                        ##Save outputs and corresponding xml in output folder
                                        with rio.open(out_img_name, "w", **out_meta) as dest:
                                            dest.write(out_image)
                                    except: 
                                        print ("something went wrong when processing {}".format(tif))
                                        pass
                                    
    #Copy across all xml files so they stay with the data for future reference
    for xml in xmlfiles:
        shutil.copyfile(xml, os.path.join(out_dir, os.path.basename(xml)))


def clip_images_to_poly_direct_single(poly_id, poly_geom, in_dir, out_dir):
    '''
    NOTE: THIS IS NOT FINISHED. TODO.
    Clips images in in_dir to an individual polygon
    This version starts with full images and assigns new name using first 10 digits
    (unlike clip_images_to_polygon_single, which crops from bbox to poly and just uses bbox names)
    thus needs to use spatial overlap rather than id to match images to polygons.
    '''
    
    ###Correct Shapefile to match projection of images based on method above
    corrected_polys = match_projections(shapefile, in_dir)
    
    ##Default is to run through all polygons in shapefile, but option to use .csv file(shape_filter) to preselect polygons
    usefilter = False
    if shape_filter != "":
        usefilter = True
        ##Can get list of polygon ids we are interested in from labels file
        ###Get unique_id for each field polygon:
        id_list = pd.read_csv(shape_filter)['unique_id']
        ##The above gives mixed strings and integers, so make list of only strings(otherwise some won't match):
        id_list = []
        for eid in id_list:
            id_list.append(str(eid))
    else:
        usefilter = False
        print ("not using filter")
    
    with fiona.open(corrected_polys) as shapefile:
        for poly in shapefile:
            poly_id = poly['properties']['unique_id']
            #print("looking at polygon with id {} \n".format(poly_id))
            ##Limit polygons processed to those in label list (if using label list)
            if (usefilter == False) or (str(poly_id) in id_list):
                #print("poly #{} matches our do list".format(poly_id))
                #polys.append(poly_id)
                poly_geom = [poly['geometry']]
                #TODO: put id and geometry in dictionary
                #clip_images_to_polygon_single(poly_id, poly_geom, in_dir, out_dir)
    
    for id in range(len(loc_id)):
        print('processing polygon #{} with id {}, starting at:{}'.format(id, loc_id[id], datetime.now()))
        #print ("For poly: minlon = {}, minlat = {}, maxlon = {}, maxlat = {}".format(poly_minlon[id], poly_minlat[id], poly_maxlon[id], poly_maxlat[id]))
        poly_bbox = box(poly_minlon[id], poly_minlat[id], poly_maxlon[id],poly_maxlat[id])
        #Set up geography for clipping functions
        geo = gpd.GeoDataFrame({'geometry': poly_bbox}, index=[0], crs=from_epsg(4326))
        #Need to reproject to coordinate system of imagery
        geo = geo.to_crs(crs=img_crs)
        geom = [json.loads(geo.to_json())['features'][0]['geometry']]
        fid = str(loc_id[id])        
        
        for index, fp in enumerate(list_files(in_dir,".tif")):
            basename = str(os.path.basename(fp)[:21]) #Are the first 21 digits always unique???
            clip_name = fid + "_" + str(os.path.basename(tif))
            out_img_name = os.path.join(out_dir, clip_name)
            if os.path.isfile(out_img_name):
                print('file already exists')
            else:
                try:
                    with rio.open(tif) as src:
                        a_band = src.read(1)
                except:
                    print ("WARNING: Could not open file {}".format(tif))
                    pass

                try:
                    with rio.open(tif) as src:
                        a_band = src.read(1)
                    out_image, out_transform = mask(src, poly_geom, crop=True)
                    out_meta = src.meta.copy()
                    #epsg_code = int(src.crs['init'][5:])
                    out_meta.update({"driver": "GTiff",
                                        "height": out_image.shape[1],
                                        "width": out_image.shape[2],
                                        "transform": out_transform,
                                        "crs": img_crs})
                    #"crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()})
                    #Save outputs and corresponding xml in output folder
                    with rio.open(os.path.join(out_dir, out_name+".tif"), "w", **out_meta) as dest:dest.write(out_image)
                except: 
                    print ("something went wrong when processing {}".format(tif))
                    pass

#clip_images_to_AOIs(aoi_file, in_dir, out_dir)
