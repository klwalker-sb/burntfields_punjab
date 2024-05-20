#!/usr/bin/env python
# coding: utf-8
# %%
import os
import rasterio as rio
from rasterio.mask import mask
import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
#For large datasets / parallel processing:
import time
import multiprocessing


### Check projections:
'''
Checks if shapefile and raster coordinate systems match
If coordinate systems do not match, reprojects shapefile to match rasters
saves new shapefile in out directory and returns path
'''
def match_projections(shapefile, raster_dir):
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


def clip_images_to_polygon_single(poly_id, poly_geom, in_dir, out_dir, mstring):
    '''
    Clips images in in_dir to an individual polygon
    assumes that images are named with first portion matching polygon id (any # of digits, followed by _)
    Uses 'String' argument to filter directory to images that contain string in case of multiple filetypes with same name
        (for example, Planet SR analyses usually use 'AnalyticMS_SR' as String)
    and matches images to polygons based on id rather than spatial overlap
    '''
    
    ## Iterate over images and process those with matching field Id:
    print("working on polygon {} \n".format(poly_id))
    num_image_list = []
    img_names = []
    for img in os.listdir(in_dir):
        if mstring in img:
            #img_fid = str(os.path.basename(img)[:10]) #This works for 10-digit field IDs, but not others
            img_fid = img[:img.index("_")] #Gets first portion of name prior to first dash; works for any number of digits
            #print(img_fid)
            if str(img_fid) == str(poly_id):
                img_names.append(img)
                try:
                    with rio.open(os.path.join(in_dir, img)) as src:
                        out_image, out_transform = mask(src, poly_geom, crop=True)
                except Exception as e:
                    print("{} with image {} \n".format(e, img))   
                    continue
                else:
                    num_digits = int(len(img_fid)+23) ##21 is hardcoded for Planet images. Sentinel are 23
                    out_name = (os.path.basename(img)[:num_digits])
                    if out_name in out_dir:
                        print ("file already exists: {}".format(out_name))
                        pass
                    else:
                        with rio.open(os.path.join(in_dir, img)) as src:
                            out_image, out_transform = mask(src, poly_geom, crop=True)
                        out_meta = src.meta
                        ##crop and mask each raster to current polygon (makes extent the same)
                        out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})
                            ##save masked raster
                        with rio.open(os.path.join(out_dir, out_name+".tif"), "w", **out_meta) as dest:
                            dest.write(out_image)
 
    ## get count of images for field polygon with this id 
    num_images = {'field_id': poly_id, 'num_images':len(img_names)}
    num_image_list.append(num_images)
    
    df1 = pd.DataFrame(num_image_list)
    print (num_image_list)
    
    ##print file with num images per field polygon: 
    ## note: a lot of the images are blanks at this point, so count is misleading.
    #outfile = os.path.join(out_dir,'num_images_per_field.txt')
    #pd.DataFrame.to_csv(df1, outfile, sep=',', na_rep='.', index=False)

    
def chunks(input, n):
    """Yields successive n-sized chunks of input"""
    for i in range(0, len(input), n):
        yield input[i:i + n]


def clip_images_to_shapefile (shapefile, shape_filter, in_dir, out_dir, mstring, num_cores):
    '''
    uses multiprocessing with available number of cores to parse polygons out to multiple workers
    '''
    
    ###Correct Shapefile to match projection of images based on method above
    corrected_polys = match_projections(shapefile, in_dir)
    
    ##Default is to run through all polygons in shapefile, but option to use .csv file(shape_filter) to preselect polygons
    usefilter = False
    if shape_filter != "":
        usefilter = True
        ##Can get list of polygon ids we are interested in from labels file
        ###Get unique_id for each field polygon:
        id_list0 = pd.read_csv(shape_filter)['unique_id']
        ##The above gives mixed strings and integers, so make list of only strings(otherwise some won't match):
        id_list = [str(eid) for eid in id_list0]
    else:
        usefilter = False
        print ("not using filter")
    
    #Get polygon list. Can be all polygons in shapefile or filtered by list of IDs in .csv
    #polys = []
    
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
                clip_images_to_polygon_single(poly_id, poly_geom, in_dir, out_dir, mstring)
    
    ''' 
    chunked_ids=list(chunks(polys, int(polys.shape[0]/numCores)+1)) #Splits input in chunks among each core
    processes=[] #Initialize the parallel processes list
    for i in np.arange(0,num_cores):
        p = multiprocessing.Process(target=clip_images_to_polygon_single(poly_id, poly_geom, in_dir, out_dir), args=([chunked_ids[i],]))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    ####OR
    p = multiprocessing.Pool(num_cores)
    num_image_list = []
    ## load the polygon shapefile and iterate through polygons
    i = 0 #to track progress only
    with fiona.open(corrected_polys) as shapefile:
        for poly in shapefile:
            poly_id = poly['properties']['unique_id']
            #print("working on polygon #{} with id {} \n".format(p, poly_id))
            ##Limit polygons processed to those in label list (if using label list)
            if (usefilter == False) or (str(poly_id) in str(id_list)):
                poly_geom = [poly['geometry']]
                p.apply_async(clip_images_to_polygon_single(poly_id, poly_geom, in_dir, out_dir), [poly])            
            i = i + 1  
    p.join() # Wait for all child processes to close.
    '''


###TO RUN-- For local testing/running only:
#match_projections(field_shapefile, in_dir)
#clip_images_to_shapefile (field_shapefile, "", in_dir, out_dir, '.tif', 4)
