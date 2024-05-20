#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pandas as pd
import shutil
import sys

#For large datasets / parallel processing:
import multiprocessing


#####INPUTS -- For local use only (declared in bash scripts for remote use):
#RmFile = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_2_process_inspection/2019v1/cloudy_scenes.txt"
#in_dir = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/MK_code/TOAR_poly_out"
#tmp_dir = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/MK_code/ALL_TOAR_cloudy"

def filter_images_chunk_old(img_list, rm_file, in_dir, tmp_dir):
    '''
    Moves images from list into alternative directory based on image id
    Currently assumes that images in directory have 11-digit polygon id folowed by 21-digit image id
    RmFile is a .txt file with a column with heading 'To_remove' containing 20-digit (atleast) image ids
    This version copies files in directory into subdirectories before parsing out to workers to avoid problems
    with changing directory size as files are moved
    THIS METHOD TAKES HOURS/DAYS TO RUN, EVEN WITH MULTIPROCESSING 
    '''
    rm_list = {}
    rm_ids = pd.read_table(rm_file)['To_remove']
    tot_removed = 0
    for id in rm_ids:
        rm_match = str(id)[:20]
        #print("looking to remove {}".format(rm_match))
        rm_list.append(rm_match)
    
    to_remove={}
    for i in img_list:
        try:
            id1 = str(os.path.basename(i)[:31])
            id_img = id1[11:]
            #print("checking {}".format(id_img))
            if str(id_img) in rm_list:
                 to_remove.append(i) 
        except IndexError:
            print ("reached end")
            break
            
    for file in to_remove:
        shutil.move(os.path.join(in_dir,file), os.path.join(tmp_dir,file))
    print("moved {} images".format (len(to_remove)))
    tot_removed = tot_removed + len(to_remove)
    sys.stdout.flush()


def filter_images_chunk(rm_id, rm_file, in_dir, tmp_dir):
    '''
    Moves images from list into alternative directory based on image id
    Currently assumes that images in directory have 11-digit polygon id folowed by 21-digit image id
    RmFile is a .txt file with a column with heading 'To_remove' containing 20-digit (atleast) image ids
    This version checks whole directory for each id in the remove file and RUNS ORDERS OF MAGNITUDE FASTER!!!
    takes <5min for 100,000 file directory and 200 remove files (vs. many hours for first method)
    '''
    removed = 0
    rm_match = str(rm_id)[:20]
    print("looking to remove {}".format(rm_match))
    for file in os.listdir(in_dir):
        if rm_match in file:
            shutil.move(os.path.join(in_dir,file), os.path.join(tmp_dir,file))
            removed = removed + 1
    print("removed {} files". format(removed))
    sys.stdout.flush()


def filter_images(rm_file, in_dir, tmp_dir, num_cores):
    '''
    Divides directory into chunks to process over multiple cores
    this is less necessary with the current code, which runs pretty quickly on its own.
    '''
    p = multiprocessing.Pool(num_cores)
    
    rm_ids = pd.read_table(rm_file)['To_remove']
    for id in rm_ids:
        p.apply_async(filter_images_chunk(id, rm_file, in_dir, tmp_dir), [id]) 
        
    p.close()
    p.join() # Wait for all child processes to close.
    #print("removed {} blank files TOTAL from ".format (TotRemoved, out_dir))
    #print("moved {} images TOTAL into directory {}".format (TotRemoved, tmp_dir))

###For local running/testing only
#filter_images(RmFile, in_dir, tmp_dir, 4)

