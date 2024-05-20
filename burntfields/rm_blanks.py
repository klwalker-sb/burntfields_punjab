#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import rasterio as rio
import sys
#For large datasets / parallel processing:
import multiprocessing

##############TODO: fix parallel processing. For now this works, but only in series.
#####INPUTS -- For local use only (declared in bash scripts for remote use):
#out_dir = 

#even if footprints overlap, overlapping portion may contain no data. This filters out nodata returns.
def rm_blanks_chunk(flist, endstring, out_dir):
    blankfiles = []
    blankmasks = []
    lonelymeta = []
  
    for i, f in enumerate(flist):
        thisfile = os.path.join(out_dir, f)
        if endstring in f:
            try:
                with rio.open(thisfile) as check_Src:
                    a_band = check_src.read(1)
            except IndexError:
                print ("not a multiband raster")
                blankfiles.append(f)
            except:
                print ("problem opeining raster{}".format(f))
                pass
            else:
                with rio.open(thisfile) as check_src:
                    a_band = check_src.read(1)
                in_max = np.nanmax(a_band)
                if in_max == 0:
                    blankfiles.append(f)
            
    #Delete associated masks and metadata
    for bf in blankfiles:
        imgname = str(os.path.basename(bf)[:21])
        for fl in os.listdir(out_dir):
            if imgname in fl:
                if 'udm' in fl:
                    blankmasks.append(fl)
                if fl.endswith('.xml'):
                    lonelymeta.append(fl)
        os.remove(os.path.join(out_dir,bf))
    for bm in blankmasks:
        os.remove(os.path.join(out_dir,bm))
    for m in lonelymeta:
        os.remove(os.path.join(out_dir,m))
        
    print("removed {} blank raster files, {} associated masks and {} metadata files".format(len(blankfiles), len(blankmasks), len(lonelymeta)))
    num_removed = len(blankfiles) + len(blankmasks) + len(lonelymeta)

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def rm_blanks(out_dir, endstring, num_cores):
    '''
    Divides directory into chunks to process over multiple cores (otherwise, runs for days...)
    '''
    p = multiprocessing.Pool(num_cores)
    
    file_list = []
    num_files = len(os.listdir(out_dir))
    for f in os.listdir(out_dir):
        file_list.append(f)
        
    chunksize = int(num_files//num_cores)
    dataslice = chunks(file_list, chunksize)

    print("dividing into {} batches of {}".format(num_cores, chunksize))
    for b in range(num_cores):
        p.apply_async(rm_blanks_chunk(dataslice[b], endstring, out_dir), [b])   
                
    p.close()
    p.join() # Wait for all child processes to close.
    #print("removed {} blank files TOTAL from ".format (TotRemoved, out_dir))

###TO RUN-- For local testing/running only:
#rm_blanks(out_dir, '.tif', 1)