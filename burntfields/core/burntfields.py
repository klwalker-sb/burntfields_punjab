#!/usr/bin/env python
import argparse

from burntfields.clip_images_to_bbox import clip_images_to_AOIs
from burntfields.clip_images_to_shapefile import clip_images_to_shapefile
from burntfields.split_csv_file import split_csv
from burntfields.rm_blanks import rm_blanks
from burntfields.rm_images import filter_images
from burntfields.cloud_compare import list_cloud_comparisons
from burntfields.apply_masks import mask_images
from burntfields.pixel_calcs import get_all_pixel_calcs
from burntfields.rf_model import loocv, bootstrap_holdout

def main():
    parser = argparse.ArgumentParser(description='data prep for planet spectral analyses of burnt fields',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='process')

    available_processes = ['version',
                            'clip_images', 
                            'clip_to_poly', 
                            'split_file', 
                            'rm_blanks', 
                            'rm_images', 
                            'compare_clouds', 
                            'mask',
                            'pixel_calcs',
                            'bootstrap',
                            'loocv'
                            ]

    for process in available_processes:
        subparser = subparsers.add_parser(process)
         
        if process == 'version':
            continue

        subparser.add_argument('--out_dir', dest='out_dir', help='out directory for processed output images', default=None)
        subparser.add_argument('--in_dir', dest='in_dir', help='directory containing input images to process', default=None)
     
        if process == 'rm_blanks':
            subparser.add_argument('--endstring', dest='endstring', 
                                   help='Type of file to look for. If clipped, change to AnalyticMS_clip.tif', default='AnalyticMS')     
            subparser.add_argument('--num_cores', dest ='num_cores', 
                                    help='Number of cores to make available, for parallel processing', default=4, type=int)

        elif process == 'split_file':
            subparser.add_argument('--aoi_file', dest='aoi_file', help='file containing bounding box coordinates for fields', default=None)
            subparser.add_argument('--nrows', dest='nrows', help='number of rows for each file', default=100)                         
         
        elif process == 'compare_clouds':
            continue
             
        elif process == 'clip_images':
            subparser.add_argument('--aoi_file', dest='aoi_file', help='file containing bounding box coordinates for fields', default=None)
         
        elif process == 'clip_to_poly':
            subparser.add_argument('--shapefile', dest='shapefile', help='Shapefile contining polygons to clip to')
            subparser.add_argument('--shape_filter', dest='shape_filter', 
                                   help='optional file with polygon ids if only some to be used', default=None)
            subparser.add_argument('--num_cores', dest ='num_cores', 
                                   help='Number of cores to make available, for parallel processing', default=4, type=int)
            subparser.add_argument('--mstring', dest ='mstring', 
                                   help='Sting common to images to be clipped (and not other files that may be in directory')
                        
        elif process == 'rm_images':
            subparser.add_argument('--rmfile', dest='rmfile', 
                                   help='.txt file with list of image ids to remove, under header: To_remove')     
            subparser.add_argument('--num_cores', dest ='num_cores', 
                               help='Number of cores to make available, for parallel processing', default=4, type=int)
                     
        elif process == 'mask':
            subparser.add_argument('--endstring', dest='endstring', 
                                   help='File type for images to be masked. Relevant endstrings: "AnalyticMS.tif", "toar.tif", "udm.tif", "udm2.tif", "metadata.xml" or may contain clip', default='AnalyticMS_SR_clip.tif')
            subparser.add_argument('--out_type', dest='out_type', 
                                   help='Mask type. current options are: "udm"(for original unusable data mask) "udm2all" (for all bands of udm2) "udm2cs" (for cloud and shadow bands of udm2), "udm2csorig"(for cloud, shoadow and original udm)', default='udm2csorig')
         
        elif process == 'pixel_calcs':
            subparser.add_argument('--field_list', dest='field_list', help='file containing ids for fields to calculate', default=None)
            subparser.add_argument('--num_bands', dest='num_bands', 
                                help='number of bands for image type (BASMA=3, PlanetScope=4, Sentinel=9)', default=3)
            subparser.add_argument('--var_path', dest='var_path', help='path to .csv file containing list of feature variables to use', default=None)
             
        elif process == 'bootstrap':
            subparser.add_argument('--field_list' , dest='field_list', help='path to file containing ids for labeled fields to calculate')
            subparser.add_argument('--training_path' , dest='training_path', help='path to file containing training set')
            subparser.add_argument('--variable_path' , dest='variable_path', help='path to csv file containing list of variables to use')
            subparser.add_argument('--num_rep' , dest='num_rep', help='number of times to repeat bootstrap, or number of folds if k-fold.')
            subparser.add_argument('--method' , dest='method', help='boot for bootstrap, k for k-fold', default='boot')
            subparser.add_argument('--fit' , dest='fit', help='whether to fit probabilities for full set', default=False)
            subparser.add_argument('--all_features' , dest='all_features', help='all features to fit, it fit == True', default=None)
            subparser.add_argument('--seed1', dest='seed1', help='seed ', default=8)
            subparser.add_argument('--seed2', dest='seed2', help='second seed...', default=6888)
            subparser.add_argument('--drop_border' , dest='drop_border', help='whether to drop border pixels', default=True)
            subparser.add_argument('--strat' , dest='strat', help='name of column to stratify on, if stratifying k-fold', default=None)
     
        elif process == 'loocv':
            subparser.add_argument('--labeled_list' , dest='labeled_list', 
                                    help='path to file containing ids for labeled fields to calculate')
            subparser.add_argument('--ho_list' , dest='ho_list', help='path to file contiaining holdout set (at field level)')
            subparser.add_argument('--variable_path' , dest='variable_path', help='path to csv file containing list of variables to use')
            subparser.add_argument('--all_features' , dest='all_features', help='all features to fit, it fit == True', default=None)
            subparser.add_argument('--seed1', dest='seed1', help='seed ', default=8)
            subparser.add_argument('--seed2', dest='seed2', help='second seed...', default=6888)
            subparser.add_argument('--drop_border' , dest='drop_border', help='whether to drop border pixels', default=True)

    args = parser.parse_args()
     
    if args.process == 'version':
        print(__version__)        
        return
     
    elif args.process == 'clip_images':
        clip_images_to_AOIs(aoi_file = args.aoi_file, 
                             in_dir = args.in_dir, 
                             out_dir = args.out_dir)
         
    elif args.process == 'clip_to_poly':
        clip_images_to_shapefile(shapefile = args.shapefile, 
                                 shape_filter = args.shape_filter,
                                 in_dir = args.in_dir, 
                                 out_dir = args.out_dir,
                                 mstring = args.mstring,
                                 num_cores = args.num_cores)
         
    elif args.process == 'mask':
        mask_images(in_dir = args.in_dir, 
                     out_dir = args.out_dir,
                     endstring = args.endstring,
                     out_type = args.out_type)
     
    elif args.process == 'split_files':
        split_csv(aoi_file = args.aoi_file,  
                             out_dir = args.out_dir,
                             nrows = args.nrows)
                                
    elif args.process == 'compare_clouds':
        list_cloud_comparisons (in_dir = args.in_dir,
                                out_dir = args.out_dir)

    elif args.process == 'rm_blanks':
        rm_blanks(out_dir = args.out_dir,
                    endstring = args.endstring,
                   num_cores = args.num_cores)
             
    elif args.process == 'rm_images':
        filter_images(rmfile = args.rmfile,
                        in_dir = args.in_dir,
                        tmp_dir = args.out_dir,
                        num_cores = args.num_cores)
             
    elif args.process == 'calc_indices':
        refl_2_ndvi_and_bai(in_dir = args.in_dir,
                         out_dir = args.out_dir, 
                         num_cores = args.num_ores)
     
    elif args.process == 'pixel_calcs':
        get_all_pixel_calcs(field_list = args.field_list,
                           data_dir = args.in_dir,
                           out_dir = args.out_dir,
                           num_bands = args.num_bands,
                           var_path = args.var_path)

    elif args.process == 'bootstrap':
        bootstrap_holdout(out_dir = args.out_dir,
                           field_list = args.field_list,
                           training_path = args.training_path,
                           variable_path = args.variable_path,
                           num_rep = args.num_rep,
                           method = args.method,
                           fit = args.fit,
                           all_features = args.all_features,
                           seed1 = args.seed1,
                           seed2 = args.seed2,
                           drop_border = args.drop_border,
                           strat = args.strat)
      
    elif args.process == 'loocv':
        loocv(out_dir = args.out_dir,
                           labeled_list = args.labeled_list,
                           ho_list = args.ho_list,
                           variable_path = args.variable_path,
                           all_features = args.all_features,
                           seed1 = args.seed1,
                           seed2 = args.seed2,
                           drop_border = args.drop_border)

if __name__ == '__main__':
    main()
