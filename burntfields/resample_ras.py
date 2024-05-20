#!/usr/bin/env python
# coding: utf-8

## for resampling only:
import os
from osgeo import gdal
## for clipping plus resampling method:
import numpy as np
import rasterio as rio
#from rasterio import features
#from rasterio.mask import mask



#Not currently using because option below also clips rasters to polygon in same step.
def resample_ras(ras, out_dir, refdata):
    '''
    Resamples images to resolution of reference image using only gdal. 
    Does not clip the images
    '''

    #open reference file and get resolution
    reference = gdal.Open(refdata, 0)  # this opens the file in only reading mode
    referenceTrans = reference.GetGeoTransform()
    x_res = referenceTrans[1] # GeoTransform[1] is w-e pixel resolution / pixel width
    y_res = referenceTrans[5]  # GeoTransform[1] is n-s pixel resolution / pixel height (will be - if N is up; make +)

    # specify input and output filenames
    basename = str(os.path.basename(ras).strip('.tif'))
    output_file = os.path.join(out_dir,basename+'_3m.tif')

    # call gdal Warp
    kwargs = {"format": "GTiff", "xRes": x_res, "yRes": y_res}
    ds = gdal.Warp(output_file, ras, **kwargs)


def resample_ras_w_clip(Ras, output_file, bounds, dst_crs, dst_res=(3, 3), interp=0, align=True):
    '''
    Resamples and clips a raster to resolution and size of reference raster. (matches projection as well in case of missmatch)
    
    Note resample algorithm is nearest neighbor (0) by default. Changed to Bilinear here (1). Can also use Cubic (2), etc
    For WarpOptions parameters see: https://gdal.org/python/osgeo.gdal-module.html)
    maybe add to gdal.Warp: srcNodata = , dstNodata = 
     
    '''

    opt = gdal.WarpOptions(dstSRS=dst_crs,
                           outputBounds=bounds,
                           polynomialOrder=interp,
                           resampleAlg=1,
                           targetAlignedPixels=align,
                           xRes=dst_res[0], yRes=dst_res[1])
    gdal.Warp(srcDSOrSrcDSTab=Ras, 
              destNameOrDestDS=output_file, 
              options=opt)


def summarize_tif(path):
    '''
    Get quick summary of raster info; only using for data checking
    '''
    tif1 = rio.open(path)
    x_res = (tif1.bounds.right - tif1.bounds.left) / tif1.width
    y_res = (tif1.bounds.top - tif1.bounds.bottom) / tif1.height
    noData = tif1.nodata
    print("crs {}".format(tif1.crs))
    print('dtype {}'.format(tif1.meta['dtype']))
    print("height {}, width {}".format(tif1.height, tif1.width))
    print("x_res {}, y_res {}".format(x_res, y_res))
    print("bounds {} \n \n".format(tif1.bounds))
    print("no data {}".format(noData))
    tif1.close()


## For data checking only:
#summarize_tif(refdata)
#ras = 
#summarize_tif(ras)



def resample_ras_directory(in_dir, out_dir, ref_dir):
    '''
    Resamples and clips rasters in directory to resolution and size of reference raster.
    Assumes that rasters in in_dir have match in ref_dir that share the same id (first characters prior to _ in file name

    TODO: apply mask to set masked pixels to noData as in Planet data.
    
    '''
    for i in os.listdir(in_dir):
        ras = os.path.join(in_dir,i)
        basename = str(os.path.basename(ras).strip('.tif'))
        img_fid = i[:i.index("_")] 
        id_matches = []
        for f in os.listdir(ref_dir):
            if f.startswith(img_fid):
                id_matches.append(f)
        if len(id_matches) == 0:
            print("There are no matches for id {}".format(img_fid))
            pass
        else:
            refdata = os.path.join(ref_dir,id_matches[0])
            print('refdata = {}'.format(refdata))
        ## To resample only:
        #resample_ras(ras, out_dir, refdata)
        
        ##To resample and clip:
        with rio.open(refdata, masked=True) as refras:
            ref_crs = refras.crs
            #print("Images are in:", ref_crs)
            ref_bounds = tuple(refras.bounds) # (minX, minY, maxX, maxY)

        output_file = os.path.join(out_dir,basename+'clip.tif')

        if os.path.isfile(output_file):
            os.remove(output_file)
        resample_ras_w_clip(ras, output_file, bounds=ref_bounds,
               dst_crs=ref_crs, dst_res=(3, 3), interp=0, align=False)

#resample_ras_directory(in_dir, out_dir, ref_dir)

