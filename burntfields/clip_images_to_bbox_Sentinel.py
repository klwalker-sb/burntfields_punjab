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


def clip_images_to_aois_sentinel(aoi_file, in_dir, out_dir, composite=False):
    '''
    Compares polygon bounding box coordinates (in AOI_file) to Sentinel footprints to check for overlap.
    This is currently hardcoded to specific project which only overlaps three Sentinel footprints
    or can use composite=True if images are already composited and overlap is assumed (as with BASMA)
    and AOI file containing fields with headings: 'unique_id', 'minLat', 'minLon', 'maxLat', 'maxLon'
    Also designed to output IDs with structure similar to those created for Planet data
    '''
    #Check coordinate system of images:
    first_img = os.listdir(in_dir)[0]
    if first_img.endswith('tif'):
        first_path = os.path.join(in_dir, first_img)
        with rio.open(first_path) as samp_img:
            img_crs = samp_img.crs
            print("Images are in:", img_crs)
    else: 
        print('{} is not a .tif'.format(first_img))
        
    locations = pd.read_csv(aoi_file)
    loc_id = locations['unique_id']
    #loc_id = locations['poly_id']
    poly_minlon = locations['minLon'].values.tolist()
    poly_minlat = locations['minLat'].values.tolist()
    poly_maxlon = locations['maxLon'].values.tolist()
    poly_maxlat = locations['maxLat'].values.tolist()

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
        
        for f in os.listdir(in_dir):
            if f.endswith('.tif'):
                fp = str(f[:6])
                useimg=True
                if composite==False:
                    if fp == 'T43RDP':
                        img_minlon = 73.995 
                        img_maxlat = 30.729
                        img_maxlon = 75.101
                        img_minlat = 29.742
                    elif fp == 'T43REP':
                        img_minlon = 75
                        img_maxlat = 30.733
                        img_maxlon = 76.135
                        img_minlat = 29.737
                    elif fp == 'T43RDQ':
                        img_minlon = 73.945
                        img_maxlat = 31.631
                        img_maxlon = 75.102
                        img_minlat = 30.645
                    else:
                        print ('Problem getting image footprint; do not have data for scene {}.'.format(fp))
        
            #check if raster overlaps polygon (ignore if it doesn't):
                    if (img_maxlat < poly_minlat[id] 
                        or img_maxlon < poly_minlon[id] 
                        or img_minlon > poly_maxlon[id] 
                        or img_minlat > poly_maxlat[id]):
                        #Intersection = Empty
                        useimg=False
                        pass
                if useimg==True:
                    basename = str(f[:22])
                    base_id = str(basename[-15:])
                    print ("found an intersection with image {}".format(base_id))
                    #Intersection = NotEmpty
                    #If there is overlap, get corresponding .tif files and crop them to poly bbox
                    clip_name = fid + "_" + base_id + "_" + fp + '.tif'
                    print(clip_name)
                    img_in = os.path.join(in_dir, f)
                    out_img_name = os.path.join(out_dir, clip_name)
                    if os.path.isfile(out_img_name):
                        print('file already exists')
                    else:
                        try:
                            with rio.open(img_in) as src:
                                a_band = src.read(1)
                        except:
                            print ("WARNING: Could not open file {}".format(img_in))
                            pass
                        else:
                            try:
                                with rio.open(img_in) as src:
                                    a_band = src.read(1)
                                    out_image, out_transform = mask(src, geom, crop=True)
                                    out_meta = src.meta.copy()
                                    #epsg_code = int(src.crs['init'][5:])
                                    out_meta.update({"driver": "GTiff",
                                        "height": out_image.shape[1],
                                        "width": out_image.shape[2],
                                        "transform": out_transform,
                                        "crs": img_crs})
                                        #"crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()})
                                
                                ##Save outputs in output folder
                                with rio.open(out_img_name, "w", **out_meta) as dest:
                                    dest.write(out_image)
                            except: 
                                print ("something went wrong when processing {}".format(img_in))
                                pass


###Local commands
#aoi_file = 
#out_dir = 
#in_dir = 
#clip_images_to_aois_sentinel(aoi_file, in_dir, out_dir, composite=False)


