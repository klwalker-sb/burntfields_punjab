#!/usr/bin/env python
# coding: utf-8

import os
import csv
import pandas as pd
import numpy as np
import random
import sklearn
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SequentialFeatureSelector



#####STEP0_DATAPREP###########################################################################################
##############################################################################################################


def label_training_pixels(field_list, pixeldf, out_dir, drop_border=True):
    '''
    Preps output from PixelCalcs.Get_All_Pixel_Calcs() (with or without PixelCalcs.AddGeographicInfo()):
        Drops border pixels from dataset (If geographic data included and dropBorder=True in arguments)
        Drops rows with any NANs (Random forest cannot run with NANs)
        separates training (labeled) and nontraining sets
        gets label column from fields dataframe and adds to dataframe for training pixels
    '''
    field_data = pd.read_csv(field_list, usecols=['unique_id','pv_burnt_any'])
    #print(FieldData.dtypes)
    field_data['unique_id'] = field_data['unique_id'].apply(str)
    #print(FieldData.dtypes)
    pixel_data = pd.read_csv(pixeldf)
    
    ##Droping border pixels:
    if drop_border == True:
        pixel_data = pixel_data[pixel_data.border != 1]
        print('{} pixels after dropping borders'.format(len(pixel_data)))
    ##Droping pixels with NAN:
    pixel_data = pixel_data.dropna(axis=0, how='any', inplace=False)
    print('{} pixes after dropping NAs'.format(len(pixel_data)))
    
    ##Separating training(labeled) and non-training(no label) data:
    pixel_data['field_id'] = pixel_data['pixel_id'].str[:10]
    training_pixels = pixel_data
    training_pixels = pixel_data[pixel_data.field_id.isin(field_data['unique_id'])]
    training_pixels['set'] = 'TRAINING'
    #print(len(training_pixels))
    non_training_pixels = pixel_data[~pixel_data.field_id.isin(field_data['unique_id'])]
    non_training_pixels['set'] = 'NONTRAINING'
    #print(len(non_training_pixels))

    ##Getting burn labels for training(labeled) pixels:
    training_labeled = training_pixels.merge(field_data,how='left',left_on='field_id', right_on='unique_id')
    training_labeled.rename(columns={'pv_burnt_any': 'label'}, inplace=True)
    training_labeled = training_labeled.drop('unique_id', axis=1)
    #print(len(training_labeled))
    
    ##Printing labeled and no-labeles sets to csv:
    training_pix_labeled_path = os.path.join(out_dir,'pixelData_Labeled_V4mNH.csv')
    pd.DataFrame.to_csv(training_labeled, training_pix_labeled_path, sep=',', na_rep='NaN', index=False)
    non_training_pix_path = os.path.join(out_dir,'NonTrainingPixx.csv')
    pd.DataFrame.to_csv(non_training_pixels, non_training_pix_path, sep=',', na_rep='NaN', index=False)
    
    return(training_pix_labeled_path, non_training_pix_path)


def generate_holdout(field_list, out_dir, seed=8):
    '''
    Randomly samples 20% of training set (labeled data) and writes subset to file to use as holdouts
    '''
    fields = pd.read_csv(field_list)
    random.seed(seed)
    field_samp = random.sample(range(len(fields)),int((.2*len(fields))))
    field_subset = []
    for f in field_samp:
        field_id = fields['unique_id'][f]
        field_subset.append(field_id)  
    #print(field_subset)
    field_holdouts = fields[fields.unique_id.isin(field_subset)]
    field_holdouts['set']='HOLDOUT'
    field_training = fields[~fields.unique_id.isin(field_subset)]
    field_training['set']='TRAINING'
    
    ##Writing holdout and training sets to file for future use:
    holdout_field_path = os.path.join(out_dir,'V2_Fields_HoldoutSet1x.csv')
    pd.DataFrame.to_csv(field_holdouts, holdout_field_path, sep=',', na_rep='NaN', index=True)
    pd.DataFrame.to_csv(field_training, os.path.join(out_dir,'PixelData_labeled_toTrain_V4mXx.csv'), sep=',', na_rep='NaN', index=True)
    
    return (holdout_field_path)


def separate_holdout(holdout_field_path, training_pix_path, out_dir):
    '''
    Generates separate pixel databases for training data and 20% field-level holdout
    Use this instead of generate_holdout() to fit a model to an exsisting holdout set
    '''
    holdout_set = pd.read_csv(holdout_field_path)
    pixels = pd.read_csv(training_pix_path)
    
    ##if there is no 'field_id' in the pixel dataset, use the following two lines (but now 'field_id' is aready in pixels)
    #holdoutSet['unique_id'] = holdoutSet['unique_id'].apply(str)
    #pixels['field_id'] = pixels['pixel_id'].str[:10]
    
    pixels_holdouts = pixels[pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_holdouts['set']='HOLDOUT'
    pixels_training = pixels[~pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_training['set']='TRAINING'

    print("original training set had {} rows. Current training set has {} rows and holdout has {} rows."
          .format(len(pixels), len(pixels_training), len(pixels_holdouts)))
    
    training_pix_path2 = os.path.join(out_dir,'V4_Model_training_FieldLevel_toTrainNH.csv')
    pd.DataFrame.to_csv(pixels_training, training_pix_path2, sep=',', na_rep='NaN', index=False)
    holdout_field_pix_path = os.path.join(out_dir,'V4_Model_testing_FieldLevel_Holdout_FullFieldx.csv')
    pd.DataFrame.to_csv(pixels_holdouts, holdout_field_pix_path, sep=',', na_rep='NaN', index=False)
    
    return(training_pix_path2, holdout_field_pix_path)


# Optional robustness check. Not used in default model.
def balance_training_set(field_list_path):
    '''
    provides option to balance training set such that the sample size of positive and negative labels match.
    does so by randomly selecting and removing positive labels
    returns a FIELD-level dataset of fields to keep in training data (unique identifier column is 'unique_id')
    This is not used in default model.
    '''
    
    #FieldListPath = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_1_input_transformed/2019_V2/labels_2019_with_bbox_V2m3.csv"
    fields = pd.read_csv(field_list_path)
    tfields = fields[fields['Set'] == 'TRAINING']
    #TFields.head(n=5)
    tfields_pos = tfields[tfields['pv_burnt_any'] == 1]
    tfields_pos.reset_index(drop=True, inplace=True)
    poslab = len(tfields_pos)
    tfields_neg = tfields[tfields['pv_burnt_any'] == 0]
    neglab = len(tfields_neg)
    reSamp = (poslab-neglab)/poslab
    print("There are {} positive labels and {} negative labels; for balanced set, need to resample positive by {}".format(poslab, neglab, "{:.2f}".format(reSamp)))

    ran_delete = random.sample(range(len(tfields_pos)),(poslab-neglab))
    remove_fields = []
    for f in ran_delete:
        field_id = tfields_pos['unique_id'][f]
        remove_fields.append(field_id)  
    (balanced_train) = tfields[~tfields.unique_id.isin(remove_fields)]

    return((balanced_train))


def prep_train_data(trainfeatures, variable_path, balanced=False, field_list=None):
    '''
    preps training set for RF model by removing labels, etc but saving info to rejoin later)
    includes 'balanced' option to balance the training set such that pos and neg lables have same sample size
    (note this will likely delete a sizable chunk of data. balanced=False in default models)
    '''
    if isinstance(trainfeatures, pd.DataFrame):
        train_features = trainfeatures
    else:
        train_features = pd.read_csv(trainfeatures)
    print('There are {} training features'.format(len(train_features)))
    
    if balanced == True:
        print('balancing dataset...')
        balanced_train = balance_trainingSet(field_list)
        keep_fields = balanced_train['unique_id'].to_list()
        train_features = train_features[train_features.field_id.isin(keep_fields)]
        print('there are now {} training features'.format(len(train_features)))

    #Get labels in separate dataset, then drop from training dataset:
    train_labels = train_features['label']
    train_features = train_features.drop('label', axis=1)
    
    #Save ID columns before dropping from dataframe. All outputs will be in same order, so can rejoin later.
    if 'pixel_id' in train_features:
        ids = train_features['pixel_id']
    else:
        ids = train_features['field_id']
    fieldid = train_features['field_id']

    #Get list of variables to include in model:
    if type(variable_path) is list:
        variables = variable_path
    else:
        vars = pd.read_csv(variable_path, header=None)
        variables = vars[0].tolist()
     
    #Remove all columns not in variable list
    train_features = train_features[train_features.columns[train_features.columns.isin(variables)]]
    #train_features = train_features.reset_index()

    #To ensure order is maintained, save original indices of full dataset and use these for test/train split:
    indices = train_features.index.values
    #Get list of variable names, to use later:
    variable_names = (list(train_features.columns))
    
    return(train_features, train_labels, indices, ids, fieldid, variable_names)


def get_test_train(train_features, train_labels, indices):
    #Use indices of train and test rather than actual features to keep track of position in original dataset:
    x_train, x_test, indices_train, indices_test = train_test_split(train_features, indices, test_size = .01, random_state = 6888)

    print("train features shape: {}".format(x_train.shape))
    print("train indices shape: {}".format(indices_train.shape))
    print("test features shape: {}".format(x_test.shape))
    print("test indices shape: {}".format(indices_test.shape))

    #Then attach labels to indices to use as y data in model
    y_train = train_labels[indices_train]
    y_test = train_labels[indices_test]
    return (x_train, x_test, y_train, y_test, indices_train, indices_test)

#####STEP1_CREATE MODEL ######################################################################################
##############################################################################################################


def run_rf_model(x_train, y_train, seed):
    rf = RandomForestClassifier(n_estimators = 100, oob_score=True, random_state = int(seed))
    rf_model = rf.fit(x_train, y_train)
    return rf_model


#####STEP2_CHECK MODEL #######################################################################################
##############################################################################################################


def quick_accuracy (x_test, y_test, rf_model):
    predicted = rf_model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Out-of-bag score estimate: {rf_model.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
    
    cm = pd.DataFrame(confusion_matrix(y_test, predicted))
    print("Confusion Matrix: {}".format(cm))
    return accuracy, cm


def better_accuracy (rf_model, train_features, train_labels):
    ###Cross validation - -more accurate accuracy assessment:
    cv_results = cross_validate(rf_model, train_features, train_labels, cv=3, return_estimator=True)
    
    rf_fold_0 = cv_results['estimator'][0]
    print(cv_results['test_score'])
    print(cv_results['test_score'].mean())
    print(cv_results)
    
    ###seems to depend somewhat on ordering. Try this:
    #from sklearn.inspection import permutation_importance

    #result = permutation_importance(
    # rf_model, X_test, y_test, n_repeats=10, random_state=68, n_jobs=2)
    # forest_importances = pd.Series(result.importances_mean, index=feature_names)


def get_variable_importance(rf_model, x_train, out_dir):
    ### get variable importance
    importance = rf_model.feature_importances_
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(x_train.columns, rf_model.feature_importances_):
        feats[feature] = importance #add the name/value pair 

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    print(importances)
    pd.DataFrame.to_csv(importances, os.path.join(out_dir,'V2_FieldLevel_Model1_variable_importance_Refined30.csv'), sep=',', na_rep='NaN', index=True)
    return importances


def get_predprob(train_features, indices_train, indices_test, train_labels, x_train, x_test, ids, field_ids, rf_model, out_dir=None):
    train_features_predicted = train_features.copy()
    train_features_predicted.loc[indices_train,'pred_train'] = rf_model.predict_proba(x_train)[:,1]
    train_features_predicted.loc[indices_test,'pred_test'] = rf_model.predict_proba(x_test)[:,1]
    train_features_predicted['label']= train_labels
    train_features_predicted['pixel_id']=ids
    train_features_predicted['field_id']=field_ids
    
    if out_dir:
        pd.DataFrame.to_csv(train_features_predicted, os.path.join(out_dir,'V2_Model1_FieldLevel_Training_predictions.csv'), sep=',', na_rep='NaN', index=True)
    print(train_features_predicted.head(n=5))

    return train_features_predicted
    

def select_best_features(train_features,train_labels,out_dir):
    '''
    Narrows list of features to best via forward modeling (adding one feature from list at time).
    Default is 1/2 of features on list. If a different number is desired, use n_features_to_select=x in SequentialFeatureSelector(rf)
    '''
    rf = RandomForestClassifier(n_estimators = 100)
    sfs = SequentialFeatureSelector(rf, n_features_to_select=30)
    sfs.fit(train_features, train_labels)
    new_feats = list(train_features.columns[sfs.get_support()])
    refined_features = pd.DataFrame(new_feats)
    pd.DataFrame.to_csv(refined_features, os.path.join(out_dir,'V2_FieldLevel_Model1_variable_importance_RefinedHalf.csv'), sep=',', na_rep='NaN', index=True)


def print_roc_curve(holdout_fields_predicted):
    '''
    Better to use precision-recall curve when classes are imbalanced.
    but precision-recall doesn't use false negatives. Need to flip 0s & 1s first to focus on the rarer no-burn class
    see: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    '''
    #burn_probs = TrainPred[:, 1]
    burn_probs = holdout_fields_predicted['pred']
    burn_index = holdout_fields_predicted['label']
    burn_auc = roc_auc_score(burn_index, burn_probs)
    print('ROC AUC=%.3f' % (burn_auc))
    burn_fpr, burn_tpr, _ = roc_curve(burn_index, burn_probs)
    #plt.plot(burn_fpr, burn_tpr, linestyle='--', label='Model2')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.show()


def get_holdout_scores (holdout_pix, variable_path, rf_model, out_dir=None):
    ##Save info for extra columns and drop (model is expecting only variable input columns)
    
    if isinstance(holdout_pix, pd.DataFrame):
        holdout_fields = holdout_pix
    else:
        holdout_fields = pd.read_csv(holdout_pix)
        
    holdout_labels = holdout_fields['label']
    h_ids = holdout_fields['pixel_id']
    h_fieldid = holdout_fields['field_id']
    print(len(holdout_fields))
    #Get list of variables to include in model:
    
    if type(variable_path) is list:
        variables = variable_path 
    else:
        vars = pd.read_csv(variable_path, header=None)
        variables = vars[0].tolist()
   
    #Remove all columns not in variable list
    holdout_fields = holdout_fields[holdout_fields.columns[holdout_fields.columns.isin(variables)]]

    ##Calculate scores
    holdout_fields_predicted = rf_model.predict_proba(holdout_fields)
    
    ##Add extra columns back in
    holdout_fields['pred']= holdout_fields_predicted [:,1]
    holdout_fields['label']= holdout_labels
    holdout_fields['pixel_id']=h_ids
    holdout_fields['field_id']=h_fieldid

    if out_dir:
        pd.DataFrame.to_csv(holdout_fields, os.path.join(out_dir,'V3_Model3_FieldHoldouts_predictions.csv'), sep=',', na_rep='NaN', index=True)
   
    print(holdout_fields.head(n=5))
    return holdout_fields
    

def get_holdout_accuracy(holdout_df):
    n_truepos = int(len(holdout_df[(holdout_df.pred >= .5) & (holdout_df.label == 1)]))
    n_falsepos = int(len(holdout_df[(holdout_df.pred >= .5) & (holdout_df.label == 0)]))
    n_missedpos = int(len(holdout_df[(holdout_df.pred < .5) & (holdout_df.label == 1)]))
    n_trueneg = int(len(holdout_df[(holdout_df.pred < .5) & (holdout_df.label == 0)]))
    oa = (n_truepos + n_trueneg)/(n_truepos+n_falsepos+n_missedpos+n_trueneg)
    
    print(oa)
    return oa, n_truepos, n_falsepos, n_missedpos, n_trueneg


##############FOR PLANET-SENTINEL COMBO
def check_coord_match(planet_path, sentinel_path, out_dir):
    ##CheckCoordinates of datasets to merge:
    planet_df = pd.read_csv(planet_path, usecols=['pixel_id','CoordX','CoordY'], index_col=0)
    sentinel_df = pd.read_csv(sentinel_path, usecols=['pixel_id','CoordX','CoordY'], index_col='pixel_id')
    sentinel_df.rename(columns={'CoordX':'CoordX_S', 'CoordY':'CoordY_S'}, inplace=True)
    coord_check_df = planet_df.merge(sentinel_df, how='inner', left_index=True, right_index=True)

def merge_planet_sentinel(planet_path, sentinel_path, use_sentinel, out_dir):
    ##Merge planet and Sentinel Datasets
    planet_df = pd.read_csv(planet_path, index_col='pixel_id')
    sentinel_df = pd.read_csv(sentinel_path, usecols=use_sentinel, index_col='pixel_id')
    sentinel_df.rename(columns={'imgCount':'numSentinel', 'red20':'red20_S', 'blueAvg':'blueAvg_S', 'redDrop1':'redDrop1_S',
                              'CIMin':'CIMin_S', 'CIDrop1':'CIDrop1_S', 'blueMax':'blueMax_S', 'nirMin':'nirMin_S', 
                               'nirDrop1':'nirDrop1_S', 'nirAvg':'nirAvg_S', 'greenMax':'greenMax_S'}, inplace=True)
    pixeldf_all = planet_df.merge(sentinel_df, how='inner', left_index=True, right_index=True)
    ###Save as final pixel dataframe to be used in analyses
    pd.DataFrame.to_csv(pixeldf_all, os.path.join(out_dir,'pixelData_combined.csv'), sep=',', na_rep='NaN', index=True)
    
def merge_planet_basma(planet_path, basma_path, use_basma, out_dir):    
    planet_df = pd.read_csv(planet_path, index_col='pixel_id')
    basma_df = pd.read_csv(basma_path, usecols=use_basma, index_col='pixel_id')
    basma_df.rename(columns={'imgCount':'numBASMA', 'blueAvg':'BAS_GVAvg', 'blueMax':'BAS_GVMax', 'blueStdv':'BAS_GVStdv',
                            'greenAvg':'BAS_SAvg', 'greenMax':'BAS_SMax', 'greenStdv':'BAS_SStdv', 'greenMin':'BAS_SMin', 
                            'redAvg':'BAS_CharAvg', 'redMax':'BAS_CharMax', 'redStdv':'BAS_CharStdv', 'redMin':'BAS_CharMin',
                            'redSpike0':'BAS_CharSpike0', 'greenSpike0':'BAS_SSpike0', 'greenDrop0':'BAS_SDrop0'}, inplace=True)
    pixeldf_all = planet_df.merge(basma_df, how='inner', left_index=True, right_index=True)
    print('final DF has {} rows'.format(len(pixeldf_all)))
    pd.DataFrame.to_csv(pixeldf_all, os.path.join(out_dir,'pixelData_combined_wBASMA.csv'), sep=',', na_rep='NaN', index=True)


#####STEP3_FIT MODEL TO NONTRAINING DATA #####################################################################
##############################################################################################################

def get_all_scores (feature_path, variable_path, rf_model, out_dir=None, print_db=False):
    '''
    Gets continuous scores (/predictions) for all features (pixels or fields)
    This works for both pixel-level and field-level models by setting pixlev to T/F based on whether a pixel id is detected.
    '''
    ##Save info for extra columns and drop (model is expecting only variable input columns)
    if isinstance(feature_path, pd.DataFrame):
        features = feature_path
    else:
        features = pd.read_csv(feature_path)
    
    pixlev=False
    
    if 'pixel_id' in features:
        pixlev=True
        n_ids = features['pixel_id']
        unique_ids = features.pixel_id.unique()
        print('{} unique pixel ids for {} rows in db'.format(len(unique_ids), len(features)))
    else:
        n_ids = features['field_id']
        print('{} field-level ids'.format(len(features)))
        
    if 'field_id' in features.columns:
        n_fieldid = features['field_id']
    else:
        features['field_id'] = features['pixel_id'].str[:10]
        n_fieldid = features['field_id']
    
    #Get list of variables to include in model:
    vars = pd.read_csv(variable_path, header=None)
    variables = vars[0].tolist()
    print('model has {} variables'.format(len(variables)))
    #Remove all columns not in variable list
    features = features[features.columns[features.columns.isin(variables)]]
    print(features.shape)
    features = features.dropna(how='any',axis=0)
    print(features.shape)
    
    ##Calculate scores
    features_predicted = rf_model.predict_proba(features)
    ##Add extra columns back in
    features['pred']= features_predicted [:,1]
    if pixlev==True:
        features['pixel_id']=n_ids
    features['field_id']=n_fieldid

    if print_db == True:
        pd.DataFrame.to_csv(features, os.path.join(out_dir,'V3_Model1_predictions.csv'), sep=',', na_rep='NaN', index=True)
    
    print(features.head(n=5))
    return features
    

def simplify_dataset(df_in, out_dir):
    '''
    chops unnecessary columns off of pixel-level df to output only prediction scores with ids 
    to reduce size of dataset
    '''
    keep_variables = ['pixel_id','field_id','pred']
    df1 = df_in[df_in.columns.intersection(keep_variables)]
    print(len(df1))
    
    ##Can print here, but redundant if doing get_final_pixel_dataset()
    if out_dir:
        pd.DataFrame.to_csv(df1, os.path.join(out_dir,'V2_pixel_predictions_short.csv'), sep=',', na_rep='NaN', index=True)
    
    return(df1)


def split_csv_k(csv_file, out_dir, k, shuffle=True, strat=None, seed=8):
    '''
    Splits csv file into {k} separate files for k-fold analysis.
    by default, shuffles dataframe first to randomize records in each sample
    Has option to stratify sample on a column {strat} in csvFile.
    '''
    fullfile = pd.read_csv(csv_file)
    
    if shuffle == True:
        print("shuffling fields")
        ###Randomly shuffle rows before dividing
        fullfile = fullfile.sample(frac=1, random_state=seed).reset_index(drop=True)
        
    if strat:
        strats = fullfile.groupby(strat).size().reset_index(name='counts')
        fullfile['rec'] = fullfile.groupby(strat).cumcount()

        for idx, s in strats.iterrows():
            print('Working on strata: {}'.format(s['treatment']))
            ###Divide strata into k equal number of records:
            nrows = s['counts']//k
            ###If uneven remainder, need to divide between folds:
            remain = (s['counts'] % k)
            exrows = np.zeros(k)
            for r in range(remain):
                exrows[r]=1
            print('{} fields in this strata. {} fields in each fold with r {}.'.format(s['counts'],nrows,remain))
            addrows = 0
            for i in range(k):
                if i == 0:
                    startrow = 0
                else:
                    addrows = addrows + exrows[i-1]
                    startrow = nrows*i + addrows
                endrow = startrow + nrows + exrows[i] -1
                print('Fold {} has {} extra fields'.format(i,exrows[i]))
                print('pulling rows between {} and {}'.format(startrow, endrow))
                fullfile.loc[(fullfile[strat]==s[strat]) & (fullfile['rec'].between(startrow, endrow)),'HO'] = i
                
        fullfile.to_csv(os.path.join(out_dir,'kfoldcheck.csv'))
        for i in range(k):
            chunk = fullfile.loc[(fullfile['HO']==i)]
            chunk.to_csv(os.path.join(out_dir,'{}.csv'.format(i)), index=False)
    
    else:
        print("no strat")
        nrows = len(fullfile)//k +1
        for i in range(k):
            chunk = fullfile[nrows*i:nrows*(i+1)]
            chunk.to_csv(os.path.join(out_dir,'{}.csv'.format(i)), index=False)


def bootstrap_holdout(out_dir, field_list, training_path, variable_path, num_rep, method='boot', fit=False, all_features=None, seed1=8, seed2=6888, drop_border=True, strat=None):
    '''
    method = 'k' or 'boot'. 
    Runs {num_rep} RF models with a different random selection of holdouts each time.
    For method = 'boot', holdout is sampled randomly {num_rep} times, with replacement (num times a field is held out is random)
    for method = 'k', holdout and training are divided into {num_rep} partitions and each is used only once (no replacement), as in a standard k-fold
    This is used when a normal kfold run does not work because whole fields need to be held out but the unit is the pixel.
    returns a dataframe with overall accuracy and number that can be used to create accuracy matrix
    For purposes of replication: {seed1} fixes holdout sequence and {seed2} fixes rf model
    if fit == True, fits final model by getting predictions for each nontraining field 
    in each run and printing dataframes for predictions and holdout scores.
    '''
    acc = {}
    
    if fit == True:
        features = pd.read_csv(all_features)
        features_unique = features.pixel_id.unique()
        print("pixel dataset has {} rows and {} unique pixel IDs".format(len(features),len(features_unique)))
        features2 = features.drop_duplicates(subset=['pixel_id'], keep='first')
        print("pixel dataframe has {} rows after dropping duplicates".format(len(features2)))
        features2 = features2.dropna(axis=0, how='any', inplace=False)
        print("{} pixes after dropping NAs".format(len(features2)))
        if drop_border == True:
            features2 = features2[features2.border != 1]
            print("{} pixels after dropping borders".format(len(features2)))
        feature_ids = features2['pixel_id'].tolist()
        preds = []
        ho = []
        #Get full list of training pixels to frame HO index
        features2['field_id'] = features2['pixel_id'].str[:10]
        field_data = pd.read_csv(field_list, usecols=['unique_id','pv_burnt_any'])
        field_data['unique_id'] = field_data['unique_id'].apply(str)
        training_pixels = features2[features2.field_id.isin(field_data['unique_id'])]
        training_labeled = training_pixels.merge(field_data,how='left',left_on='field_id', right_on='unique_id')
        training_labeled.rename(columns={'pv_burnt_any': 'label'}, inplace=True)
        all_train = training_labeled[['pixel_id','label']]
        #all_train = all_train.set_index('pixel_id', drop=True)
        ho.append(all_train)
        print(ho)
    
    if method == 'k':
        ##Partition full field list into {num_rep} partitions to be used as holdouts for each run:
        split_csv_k(field_list, os.path.join(out_dir,'kfold'), num_rep, shuffle=True, strat=strat, seed=seed1)
            
    for i in range(num_rep):
        if method == 'k':
            print("working on fold {}...".format(i))
            #use holdouts files generated above
            holdout_fields = os.path.join(out_dir,'kfold',str(i)+'.csv')
        elif method == 'boot':
            print("working on repitition {}...".format(i))
            ##seperating random 20% holdout from training set:
            seedx = seed1 + i
            holdout_fields = generate_holdout(field_list, out_dir, seedx)
        train_hold = separate_holdout(holdout_fields, training_path, out_dir)
        ##building rf model with remaining (100-100/k)% of training data:
        train_prep = prep_train_data (train_hold[0], variable_path)
        model_parts = get_test_train(train_prep[0], train_prep[1], train_prep[2])
        rf_model = run_rf_model(model_parts[0], model_parts[2], seed2)
        ##fitting model to holdout and assessing accuracy:
        holdout_df = get_holdout_scores(train_hold[1], variable_path, rf_model, out_dir)
        acc_ho = get_holdout_accuracy(holdout_df)
        acc[i] = {}
        acc[i]['TruePos']=acc_ho[1]
        acc[i]['FalsePos']=acc_ho[2]
        acc[i]['MissedPos']=acc_ho[3]
        acc[i]['TrueNeg']=acc_ho[4]
        acc[i]['OA']=acc_ho[0] 

        if fit == True:
            scores = get_all_scores (features2, variable_path, rf_model, out_dir, print_db=False)
            pred_simp = scores['pred'].tolist()
            preds.append(pred_simp)
            
            ho_scores_s = holdout_df[['pixel_id', 'pred']]
            ho_scores_s.rename(columns={'pred': 'pred_'+str(i)}, inplace=True)
            ho_scores_si = ho_scores_s.drop_duplicates(subset=['pixel_id'], keep='first')
            ho.append(ho_scores_si)
            
    accdb = pd.DataFrame.from_dict(acc, orient='index')
    
    if fit == True:
        pred_out = pd.DataFrame(preds)
        pred_out = pred_out.transpose()
        pred_out['meanPred'] = pred_out.mean(axis=1) 
        pred_out['pixel_id'] = np.array(feature_ids)
        
        ho_out = pd.concat([d.set_index('pixel_id') for d in ho], axis=1).reset_index()
        #ho_out['meanPred'] = ho_out.mean(axis=1)
        
        pd.DataFrame.to_csv(ho_out, os.path.join(out_dir,'V4_HoldoutPred.csv'), sep=',', na_rep='NaN', index=False)
        pd.DataFrame.to_csv(accdb, os.path.join(out_dir,'V4_HoldoutAcc.csv'), sep=',', na_rep='NaN', index=False)
        pd.DataFrame.to_csv(pred_out, os.path.join(out_dir,'V4_predictions.csv'), sep=',', na_rep='NaN', index=False)
    
        return pred_out, ho_out
    else:
        return accdb


def loocv(out_dir, labeled_list, ho_list, variable_path, all_features, seed1=8, seed2=6888, drop_border=True):
    '''
    Leave-one-out cross validation
    ho_list can be same as LabeledList (eventually 1 of each labeled field is held out), 
    but can also be a subset of LabeledList  so that this can easily be parallized.
    TODO: Parallelize properly
    '''
    ##Data prep:
    features = pd.read_csv(all_features)
    features_unique = features.pixel_id.unique()
    print("pixel dataset has {} rows and {} unique pixel IDs".format(len(features),len(features_unique)))
    features2 = features.drop_duplicates(subset=['pixel_id'], keep='first')
    print("pixel dataframe has {} rows after dropping duplicates".format(len(features2)))
    features2 = features2.dropna(axis=0, how='any', inplace=False)
    print("{} pixes after dropping NAs".format(len(features2)))
    if drop_border == True:
        features2 = features2[features2.border != 1]
        print("{} pixels after dropping borders".format(len(features2)))
    features2['field_id'] = features2['pixel_id'].str[:10]
    feature_ids = features2['pixel_id'].tolist()
    
    labeled_data = pd.read_csv(labeled_list, usecols=['unique_id','pv_burnt_any'])
    labeled_data['unique_id'] = labeled_data['unique_id'].apply(str)
    training_pixels = features2[features2.field_id.isin(labeled_data['unique_id'])]
    training_labeled = training_pixels.merge(labeled_data,how='left',left_on='field_id', right_on='unique_id')
    training_labeled.rename(columns={'pv_burnt_any': 'label'}, inplace=True)
    
    preds = []
    
    ##Hold out each field. Using a seperate HO candidate dataset here from full labeled set so that this is easy to parallellize
    hodf = pd.read_csv(ho_list)
    hodf.rename(columns={'pv_burnt_any': 'label', 'unique_id':'pixel_id'}, inplace=True)
    hodf0 = hodf[['pixel_id','label']]
    ho = []
    ho.append(hodf0)
    
    for i, row in hodf0.iterrows():
        localho = []
        ho_id = row['pixel_id']
        localho.append(str(ho_id))
        print(ho_id)
        training_pix = training_labeled[~training_labeled.field_id.isin(localho)]
        training_pix['set']='TRAINING'
        holdout_pixels = training_labeled[training_labeled.field_id.isin(localho)]
        holdout_pixels['set']='HOLDOUT'
        print("original training set had {} rows. Current training set has {} rows and holdout has {} rows."
              .format(len(training_labeled), len(training_pix), len(holdout_pixels)))

        ##building rf model with remaining training data:
        train_prep = prep_train_data (training_pix, variable_path)
        model_parts = get_test_train(train_prep[0], train_prep[1], train_prep[2])
        rf_model = run_rf_model(model_parts[0], model_parts[2], seed2)
        ##fitting model to holdout and assessing accuracy:
        holdout_df = get_holdout_scores(holdout_pixels, variable_path, rf_model, out_dir)
        
        scores = get_all_scores (features2, variable_path, rf_model, out_dir, print_db=False)
        pred_simp = scores['pred'].tolist()
        preds.append(pred_simp)
            
        ho_scores_s = holdout_df[['pixel_id', 'pred']]
        ho_scores_s.rename(columns={'pred': 'pred_'+str(i)}, inplace=True)
        ho_scores_si = ho_scores_s.drop_duplicates(subset=['pixel_id'], keep='first')
        ho.append(ho_scores_si)
            
    pred_out = pd.DataFrame(preds)
    pred_out = pred_out.transpose()
    pred_out['meanPred'] = pred_out.mean(axis=1) 
    pred_out['pixel_id'] = np.array(feature_ids)
        
    ho_out = pd.concat([d.set_index('pixel_id') for d in ho], axis=1).reset_index()
    #print('len ho_out is {}'.format(length(ho_out)))
    ho_out['meanPred'] = ho_out.loc[:,[c for c in ho_out.columns if c!= "label"]].mean(axis=1)
    
    pd.DataFrame.to_csv(ho_out, out_dir,'LOOCV_HO_Pred.csv', sep=',', na_rep='NaN', index=False)
    pd.DataFrame.to_csv(pred_out, os.path.join(out_dir,'LOOCV_All_Pred.csv'), sep=',', na_rep='NaN', index=False)
    
    return pred_out, ho_out


#out_dir = "C:/Users/klobw/Desktop/Testing"
#labeled_list = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_1_input_transformed/labels_2019_with_bbox_V4.csv"
#ho_list = "C:/Users/klobw/Desktop/Testing/HOtest.csv"
#variable_path ='C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_2b_modelSpecs/V4_PixelLevel_variables_COMBO_refined49Final.csv'
#all_features = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_1_input_transformed/2019_V4/V4_pixelData_COMBO.csv"
#loocv(out_dir, LabeledList, HOlist, variablePath, AllFeatures, seed1=8, seed2=6888, dropBorder=True)


def select_variables_with_bootstrap(variable_path, switch_vars, add_vars, rm_vars, out_dir, field_list, training_path, num_rep):

    #Get list of base variables to include in model:
    base_vars = []
    with open(variable_path, newline='') as f:
        for row in csv.reader(f):
            base_vars.append(row[0])

    acc = bootstrap_holdout(out_dir, field_list, training_path, variable_path, num_rep)
    avg_acc = pd.DataFrame.mean(acc['OA'])
    print('Average Accuracy is: {}'.format(avg_acc))
    best_acc = avg_acc
    
    if switch_vars is not None:
        try_switch = pd.read_csv(switch_vars, header=None)
        for column in try_switch:
            switchies = try_switch[column].tolist()
            print('testing difference between {}'.format(switchies))
            for var in switchies:
                if var in base_vars:
                    base_vars.remove(var)
            acc_list = []
            for var in switchies:
                print ('testing variable form: {}'.format(var))
                base_vars.append(var)
                acc2 = bootstrap_holdout(out_dir, field_list, training_path, base_vars, num_rep)
                avg_acc2 = pd.DataFrame.mean(acc2['OA'])
                print('accuracy of {} is {}'.format(var, avg_acc2))
                acc_list.append(avg_acc2)
                base_vars.remove(var)
            print('final accuracy scores: '.format(acc_list))
            i_best = acc_list.index(max(acc_list))
            best_var = acc_list[i_best]
            print('best variable is {} with score of {}'.format(best_var,max(acc_list)))
            base_vars.append(var)
    
    if add_vars is not None:
        try_add = []
        with open(add_vars, newline='') as g:
            for row in csv.reader(g):
                try_add.append(row[0])
        for var in try_add:
            print ('testing additional variable: {}'.format(var))
            base_vars.append(var)
            acc2 = bootstrap_holdout(out_dir, field_list, training_path, base_vars, num_rep)
            avg_acc2 = pd.DataFrame.mean(acc2['OA'])
            if avg_acc2 < best_acc :
                base_vars.remove(var)
                print('new accuracy is {}; variable not added'.format(avg_acc2))
            else:
                best_acc = avg_acc2
                print('new accuracy is {}; variable is added'.format(best_acc))
    
    if rm_vars is not None:
        try_rm = []
        with open(rm_vars, newline='') as h:
            for row in csv.reader(h):
                try_rm.append(row[0])
        for var in try_rm:
            print ('testing removal of variabl: {}'.format(var))
            base_vars.remove(var)
            acc2 = bootstrap_holdout(out_dir, field_list, training_path, base_vars, num_rep)
            avg_acc2 = pd.DataFrame.mean(acc2['OA'])
            if avg_acc2 < best_acc :
                base_vars.append(var)
                print('accuracy not improved; variable not dropped')
            else:
                best_acc = avg_acc2
                print('new accuracy is {}; variable is dropped'.format(best_acc))
            
    final_vars = pd.DataFrame(base_vars)
    pd.DataFrame.to_csv(final_vars, os.path.join(out_dir,'FinalVars.csv'), sep=',')
    
    return final_vars


##For use with singular models (no bootstrapping) 
def get_final_pixel_dataset(all_pixels, training_pixpath, holdout_pixpath, out_dir):
    '''
    For output from single runs (not for bootstrapping outputs)
    Removes unnecessary columns as per simplify_dataset,
    Adds flagged columns to dataset to indicate TRAINING, HOLDOUT or NOLABEL
    Drops pixels for field IDs that are not in Field-level dataset (because too small)
    Also outputs list of fields with flag for data checking
    '''
    
    ##prelim data check:
    all_pixels_unique = all_pixels.pixel_id.unique()
    print('pixel dataset has {} rows and {} unique pixel IDs'.format(len(all_pixels),(len(all_pixels_unique))))
    all_pixels2 = all_pixels.drop_duplicates(subset=['pixel_id'], keep='first')
    print('pixel dataframe has {} rows after dropping duplicates'.format(len(all_pixels2)))
    
    ##Get holdout and training flags:
    holdoutdf = pd.read_csv(holdout_pixpath)
    holdoutdf.field_id = holdoutdf.field_id.astype(str)
    hold_fields = holdoutdf.field_id.unique()
    holdoutdf.unique_id = holdoutdf.unique_id.astype(str)
    train_features = pd.read_csv(training_pixpath)
    train_features.field_id = train_features.field_id.astype(str)
    train_fields = train_features.field_id.unique()
    
    ##Can get list of all unique fields in dataset for data checking
    all_fields = pd.DataFrame(all_pixels2.field_id.unique())
    all_fields['training'] = np.where(all_fields[0].isin(train_fields),1,0)
    all_fields['holdout'] = np.where(all_fields[0].isin(hold_fields),1,0)
    all_fields['holdout'] = np.where(all_fields[0].isin(holdoutdf['unique_id']),1,0)
    #pd.DataFrame(All_fields).to_csv(os.path.join(out_dir,'PixelLevel_FieldIDs.csv'), sep=',', na_rep='NaN', index=True)

    ##Simplify pixel dataset and attach flags:
    cleandf = simplify_dataset(all_pixels2, out_dir) 
    cleandf.field_id = cleandf.field_id.astype(str)
    cleandf['training'] = np.where(cleandf['field_id'].isin(train_fields),1,0)
    cleandf['holdout'] = np.where(cleandf['field_id'].isin(hold_fields),1,0)
    cleandf['holdout'] = np.where(cleandf['field_id'].isin(holdoutdf['unique_id']),1,0)
    len1 = (len(cleandf))
    
    ##Drop pixels in fields that are not in field-level dataset. Currently those are:
    ##   1115500511 - this is a 45m2 field in an urban area. Not big enough for border & interior values
    ##   1114700501 - not sure why this is getting dropped from field-level. might be fixed.  
    ##   3111000021 - not sure why this is getting dropped from field-level. Might be fixed.
    dropfield = ['1115500511', '1114700501', '3111000021']
    cleandf2 = cleandf[~cleandf['field_id'].isin(dropfield)]
    len2 = (len(cleandf2))
    print("deleted {} pixels from pixel-level data to match field-level data".format(len1-len2))
    
    ##Reattach labels
    cleandf2.set_index('pixel_id', drop=False, inplace=True)
    train_features.set_index('pixel_id', drop=False, inplace=True)
    holdoutdf.set_index('pixel_id', drop=False, inplace=True)
    labelsdf = pd.concat([train_features,holdoutdf], axis=0, ignore_index=False)

    cleandf3 = cleandf2.merge(labelsdf['label'], how="left", left_index=True, right_index=True)
    ## if no holdouts (TO DO: make holdout percentage variable and allow for 0)
    #cleandf3 = cleandf2.merge(train_features['label'], how="left", left_index=True, right_index=True)
    cleandf3['OID'] = np.arange(0, cleandf3.shape[0])
    cleandf3.set_index('OID', inplace=True)
    print(len(cleandf3))
    
    pd.DataFrame.to_csv(cleandf3, os.path.join(out_dir,'V4_RF_PixelLevel_Predictions.csv'), sep=',', na_rep='NaN', index=True)
    return cleandf3


##For use with bootstrapped model
def get_final_datasets_bootstrapped(holdoutdf, fulldf, out_dir):
    '''
    Gets average prediction for holdout data
    Adds training and holdout flags to full pixel dataset (these are the same in bootstrapped model, but are
       both included to match format of previous models)
    For training/holdout pixels, changes pred score to average score for holdouts, rather than training+holdout 
    '''
    if isinstance(holdoutdf, pd.DataFrame):
        hodf = holdoutdf
    else:
        hodf = pd.read_csv(holdoutdf, index_col=[0])
    hodf['AvgPred']= hodf[list(hodf.filter(regex='pred'))].mean(axis=1)
    pd.DataFrame.to_csv(hodf, os.path.join(out_dir,'V4_Holdout_final.csv'), sep=',', index=False)
    
    if isinstance(fulldf, pd.DataFrame):
        allpix = fulldf
    else:
        allpix = pd.read_csv(fulldf, index_col=[0])
    allpix.reset_index(inplace = True)
    allpix['field_id'] = allpix['pixel_id'].str[:10]
    ##all labeled pixels should have 1 flag for training and for holdout (holdout 1 time and training k-1 times):
    allpix['training'] = np.where(allpix['pixel_id'].isin(hodf['pixel_id']),1,0)
    allpix['holdout'] = np.where(allpix['pixel_id'].isin(hodf['pixel_id']),1,0)
    
    allpix_lab = pd.merge(allpix, hodf[['pixel_id','label','AvgPred']], how='left', on=['pixel_id', 'pixel_id'])
    allpix_lab.loc[ allpix_lab['training'] == 1, 'meanPred'] = allpix_lab['AvgPred']
    allpix_lab.rename(columns={'meanPred': 'pred'}, inplace=True)
    allpix_lab = allpix_lab.drop('AvgPred', axis=1)
    pd.DataFrame.to_csv(allpix_lab, os.path.join(out_dir,'V4_predictionsLab.csv'), sep=',', index=False)
    
    all_fields = allpix.field_id.unique()
    hodf['field_id'] = hodf['pixel_id'].str[:10]
    training_fields = hodf.field_id.unique()
    print('final model has {} fields, {} of which are training fields.'.format(len(all_fields),len(training_fields)))


##For use with LOOCV model
def get_final_datasets_loocv(holdoutdf, fulldf, labeled_list, out_dir):
    '''
    Gets average prediction for holdout data
    Adds training and holdout flags to full pixel dataset (these are the same in LOOCV model, but are
    both included to match format of previous models)
    For training/holdout pixels, changes pred score to average score for holdouts, rather than training+holdout 
    '''
    if isinstance(holdoutdf, pd.DataFrame):
        hodf = holdoutdf
    else:
        hodf = pd.read_csv(holdoutdf)    

    hodf['field_id'] = hodf['pixel_id'].str[:10]

    labeled_data = pd.read_csv(labeled_list, usecols=['unique_id','pv_burnt_any'])
    labeled_data['unique_id'] = labeled_data['unique_id'].apply(str)
    hodf_lab = hodf.merge(labeled_data,how='left',left_on='field_id', right_on='unique_id')
    hodf_lab.rename(columns={'pv_burnt_any': 'label', 'meanPred':'AvgPred'}, inplace=True)
    
    if isinstance(fulldf, pd.DataFrame):
        allpix = fulldf
    else:
        allpix = pd.read_csv(fulldf)

    allpix['meanPred'] = allpix.loc[:,[c for c in allpix.columns if c!= "pixel_id"]].mean(axis=1)
    allpixf = allpix[['pixel_id','meanPred']]
    allpixf['field_id'] = allpixf['pixel_id'].str[:10]

    ##all labeled pixels should have 1 flag for training and for holdout (holdout 1 time and training k-1 times):
    allpixf['training'] = np.where(allpixf['pixel_id'].isin(hodf_lab['pixel_id']),1,0)
    allpixf['holdout'] = np.where(allpixf['pixel_id'].isin(hodf_lab['pixel_id']),1,0)

    allpix_lab = pd.merge(allpixf, hodf_lab[['pixel_id','label','AvgPred']], how='left', on=['pixel_id', 'pixel_id'])
    allpix_lab.loc[allpix_lab['training'] == 1, 'meanPred'] = allpix_lab['AvgPred']
    allpix_lab = allpix_lab.drop('AvgPred', axis=1)
    allpix_lab.rename(columns={'meanPred': 'pred'}, inplace=True)

    pd.DataFrame.to_csv(allpix_lab, os.path.join(out_dir,'V4_LOOCV_predictionsLab.csv'), sep=',', index=False)
    hodf_lab.rename(columns={'AvgPred': 'pred'}, inplace=True)
    pd.DataFrame.to_csv(hodf_lab, os.path.join(out_dir,'V4_LOOCV_Holdout_final.csv'), sep=',', index=False)


### To run locally:
#out_dir = "C:/Users/klobw/Desktop/Testing"
#holdoutdf = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_3_output_raw/2019v5/LOOCV_Holdout_Predictions.csv"
#fulldf = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_3_output_raw/2019v5/LOOCV_PixelPredictions_Full.csv"
#labeled_list = "C:/Users/klobw/Desktop/PunjabBurning/Analysis/KW_code/burntfields/Data/Data_1_input_transformed/labels_2019_with_bbox_V5.csv"
#get_final_datasets_loocv(HoldoutDF, FullDF, LabeledList, out_dir)

def check_model_versions_pixLevel(checkpix, old_model_path, new_model):
    ##print predictions for a list of pixels to verify that values match with previous version

    old = pd.read_csv(old_model_path)
    check = old[old['pixel_id'].isin(checkpix)]
    print(check)
    check2 = new_model[new_model['pixel_id'].isin(checkpix)]
    print(check2)


######################EXPERIMENTAL############################################################################
##############################################################################################################

def get_single_pixels(pixel_features_path, field_path, variable_path, nrfs, out_dir):
    '''
    fields_path is a df containing field IDs for non-holdout training fields & 'numpix'
    variable_path is the path to a .csv file containing a list of variables used in the model (no header)
    '''
    if isinstance(pixel_features_path, pd.DataFrame):
        pixeldf = pixel_features_path
    else:
        pixeldf = pd.read_csv(pixel_features_path)
    
    fielddf = pd.read_csv(field_path)
    fields = fielddf[fielddf["Set"]=='TRAINING']
    fields.reset_index(drop=True, inplace=True)
    field_list = []

    for f in range(len(fields)):
        field_id = fields['unique_id'][f]
        pixels = pixeldf[pixeldf['field_id']==field_id]
        pix_count = len(pixels)
        field_data = [field_id, pix_count]
        field_list.append(field_data)

    #need to add cumulative pixel count to pixel_df because pixel_id is not cummulative
    pixeldf['pixid_n'] = pixeldf.groupby(['field_id']).cumcount()
    pixeldf['pixid_cum'] = pixeldf.field_id.map(str) + '_' + pixeldf.pixid_n.map(str)

    accuracy_list = []
    models_cm = []
    accuracy_out_path = os.path.join(out_dir,'V4_RFe_PixPerField_Accuracy.csv')
    confusion_out_path = os.path.join(out_dir,'V4_RFe_PixPerField_Confusion.csv')
    vidf = pd.read_csv(variable_path, index_col=0)
    vidf.index_name = 'variable'
    vi_out_path = os.path.join(out_dir,'V2_RFe_PixPerField_VariableImportance.csv')
    pred_df = pd.DataFrame(field_list, columns=('field_id','numPix'))
    pred_df.set_index('field_id',drop=True, inplace=True)
    pred_out_path = os.path.join(out_dir,'V2_RFe_PixPerField_Predictions.csv')
    
    iter_n = 0
    for iter_n in range(nrfs):
        print('running sample #{}'.format(iter_n))
        iter_n = iter_n + 1
        #for each field in field df, select 1 pixel at random
        samp_pix = []
        for fi in range(len(field_list)):
            fieldid = field_list[fi][0]
            numpix = field_list[fi][1]
            #seed ran, where?
            if(numpix)>0:
                rannum = random.randint(0,numpix)
                ranpix = str(fieldid) + "_" + str(rannum)
                samp_pix.append(ranpix)
        print("{} pixels in sample".format(len(samp_pix)))
        sample = pixeldf[pixeldf['pixid_cum'].isin(samp_pix)]
        sample_path = os.path.join(out_dir,'Sample.csv')
        pd.DataFrame.to_csv(sample, sample_path, sep=',', na_rep='NaN', index=True)
    
        sample_check = pixeldf['pixid_cum'].tolist()
        missing = np.setdiff1d(samp_pix,sample_check)
        if len(missing)>0:
            print("the following sample pixel was not used: {}".format(missing))
    
        #fit rf model using test-train split
        preppix = prep_train_data(sample_path, variable_path)
        testtrain = get_test_train(preppix[0], preppix[1], preppix[2])
        rf_model = run_rf_model(testtrain[0], testtrain[2], 5555)
    
        # get accuracy of rf model and add to big df
        acc = quick_accuracy (testtrain[1], testtrain[3], rf_model)
        accuracy_list.append(acc[0])
        cm = pd.DataFrame(acc[1])
        models_cm.append(cm)

        # get variable importance and add to big df
        imp = get_variable_importance(rf_model, testtrain[0])
        #imp.index_name = 'variable'
        #vidf.merge(imp, on='variable', how='left')
    
        #append 2nd row of importances to VI df
    
        ## get pixel prediction and add to big df
        ##use get_all_scores instead of get_predprob
        #get_all_scores (feature_path, variable_path, rf_model, out_dir, print_db=False):
        #pred = get_predprob(preppix[0], testtrain[4], testtrain[5], testtrain[0], train_labels, testtrain[1], preppix[3], preppix[4], rf_model)
        #pred_n = pred[['field_id','pred_train']]
        #preddf = preddf.merge(pred_n, on='field_id', how='left')
    
    finalcm = pd.concat(models_cm)
    accuracy_file = open(accuracy_out_path,'w')
    for element in accuracy_list:
        accuracy_file.write(str(element) + "\n")
    pd.DataFrame.to_csv(finalcm, confusion_out_path , sep=',', na_rep='NaN', index=True)
    pd.DataFrame.to_csv(vidf, vi_out_path, sep=',', na_rep='NaN', index=True)
    #pd.DataFrame.to_csv(preddf, pred_out_path, sep=',', na_rep='NaN', index=True)
