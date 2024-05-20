#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pandas as pd


def count_images (in_dir, out_dir, aoi_file):
    '''
    Returns number of images in directory for each field ID (first 10 digits of image id)
    '''
    num_image_list = []
    locations = pd.read_csv(aoi_file)
    loc_id = locations['unique_id']
    for id in loc_id: 
        images = []
        for f in os.listdir(in_dir):
            id_img = os.path.basename(f[:10])
            if str(id) == str(id_img):
                images.append(f)
        num_images = {'field_id': id, 'num_images':len(images)}
        num_image_list.append(num_images)

    df1 = pd.DataFrame(num_image_list)
    print (num_image_list)

    ##print file with num images per field polygon: 
    outfile = os.path.join(out_dir,'num_images_per_field.txt')
    pd.DataFrame.to_csv(df1, outfile, sep=',', na_rep='.', index=False)

# in_dir =
#out_dir =
#aoi_file =
#count_images(in_dir, out_dir, aoi_file)

