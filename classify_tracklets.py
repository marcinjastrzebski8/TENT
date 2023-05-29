"""
Classify tracklets in a region of a detector.
Classification model can be trained or loaded.
"""
import time
t_start = time.time()
import cProfile as profile #should also do mprof for memory usage
import pstats #this for profiling as well

import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from kernel_methods.QKE_SVC import QKE_SVC
from sliced_detector_analysis.tools import (find_detector_region as find_region, 
load_events,
parse_args,
load_config)



def data_transform(data_in_region, method = 'global') -> None:
    """
    Mean-normalise hits making up the object data (doublets, triplets)
    """
    coords = data_in_region['object_coords']
    objects_to_be_scaled = np.array([i for i in coords.values])
    hit_dim = 3
    num_objects = np.shape((objects_to_be_scaled))[0]
    num_hits_in_object = int(np.shape(objects_to_be_scaled)[1]/hit_dim)

    #make wide array into long array (cut up cols corresponding to object-hits, put back as extra rows)
    object_hits_into_rows = []
    for rank in range(num_hits_in_object):
        hits_with_current_rank = objects_to_be_scaled[:, rank*hit_dim:(rank+1)*hit_dim]
        object_hits_into_rows.append(hits_with_current_rank)
    object_array_long = np.concatenate(object_hits_into_rows)

    #consider all unique hits - this is the raw data that objects are made of
    unique_hits = np.unique(object_array_long, axis = 0)
    if method == 'global':
        max_rho = 1026
        max_phi = np.pi
        max_z = 1083
        max_vals = [max_rho, max_phi, max_z]

        object_hit_features_ready = np.divide(object_array_long, max_vals)
    elif method == 'inner_global':
        max_rho = 120
        max_phi = np.pi
        max_z = 495
        max_vals = [max_rho, max_phi, max_z]

        object_hit_features_ready = np.divide(object_array_long, max_vals)
    elif method == 'manual' or method == 'maxabs_mean':
        hit_feature_means = np.mean(unique_hits, axis = 0)
        object_hit_features_centred = np.subtract(object_array_long, hit_feature_means)
        
        if method == 'manual':
            hit_feature_mins = np.min(unique_hits, axis = 0)
            hit_feature_maxs = np.max(unique_hits, axis = 0)

            object_hit_features_ready = np.divide(object_hit_features_centred,(hit_feature_maxs - hit_feature_mins))
        else:
            unique_hits_centred = np.subtract(unique_hits, hit_feature_means)
            scaler = preprocessing.MaxAbsScaler().fit(unique_hits_centred)

            object_hit_features_ready = scaler.transform(object_hit_features_centred)
    elif (method != 'manual') and (method != 'maxabs_mean'):
        if method == 'minmax01':
            scaler = preprocessing.MinMaxScaler()
        elif method == 'minmax_to_1':
            scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
        elif method == 'minmax_to_pi2':
            scaler = preprocessing.MinMaxScaler(feature_range = (-np.pi/2, np.pi/2))
        elif method == 'maxabs':
            scaler = preprocessing.MaxAbsScaler()
        else:
            raise ValueError('Choose from: global, manual, minmax01, minmax_to_1, minmax_to_pi2, maxabs, maxabs_mean')
        scaler.fit(unique_hits)

        object_hit_features_ready = scaler.transform(object_array_long)
    #make long array back into wide array (cut up rows, put back as extra cols)
    object_hits_into_cols = []
    for rank in range(num_hits_in_object):
        hits_with_current_rank = object_hit_features_ready[rank*num_objects:(rank+1)*num_objects, :]
        object_hits_into_cols.append(hits_with_current_rank)
    object_array_wide = np.concatenate(object_hits_into_cols, axis = 1)

    objects_scaled = {'scaled_object_coords': [i for i in object_array_wide]}
    data_in_region.insert(0, 'object', objects_scaled['scaled_object_coords'])


def add_physics_to_tracklet(data_in_region, properties_to_add) -> None:

    scaler = preprocessing.MaxAbsScaler() #previously MinMax
    if not isinstance(properties_to_add, list):
         print('No physical properties added to tracklet')
         pass
    else:
        ready_properties = []
        for prop in properties_to_add:
            prop_column = data_in_region[prop]
            #need to make it a 2d array for scaler to work
            prop_to_be_scaled = [[i] for i in prop_column]
            scaler.fit(prop_to_be_scaled)
            prop_scaled = scaler.transform(prop_to_be_scaled)
            #back to 1d array
            prop_ready = [i[0] for i in prop_scaled]
            ready_properties.append(prop_ready)

        for i in range(len(data_in_region)):
            stack_of_properties = np.hstack([data_in_region.loc[i]['object'], [ready_properties[prop][i] for prop in range(len(ready_properties))]])
            data_in_region.loc[i]['object'] = stack_of_properties


if __name__ == '__main__':
    #NOTE: COMMENTED OUT PROFILING, WANT TO DO IT IN A SEPARATE BRANCH AS IT INCREASES AMOUNT OF OUTPUT PRODUCED
    #AND PROBABLY SLOWS PERFORMANCE DOWN AS WELL
    #prof = profile.Profile() #(!)
    #prof.enable() #(!)
    #load configuration parameters defining the classification run
    config, config_filename = load_config.load_config(parse_args.parse_args().config)

    #profiling_stats_filename = 'classify_tracklets_'+config['kernel_type']+'_'+str(config['num_train'])+'_'+str(config['num_test'])+'.txt' #(!)

    #profiling_stats_path = str(Path().absolute() /'profiling_stats')+'/'+profiling_stats_filename #(!)

    QKE_model = QKE_SVC(config['kernel_type'], 
    config['class_weight'], 
    gamma_class = config['gamma_class'],
    gamma_quant = config['gamma_quant'],
    C_class = config['C_class'],
    alpha = config['alpha'],
    C_quant = config['C_quant'],
    paulis = config['paulis'],
    circuit_width = config['circuit_width'],
    keep_kernel = config['keep_kernel'],
    entanglement = config['entanglement'])

    #decide detector division
    if config['division'] == 'new_phi':
        div = find_region.edge_new_phi_division()
    elif config['division'] == 'z':
        div = find_region.edge_z_division()
    elif config['division']== 'layer':
        div = find_region.edge_phi_division()
    elif config['division'] == 'phi':
        div = find_region.edge_phi_division()
    elif config['division'] == 'eta':
        div = find_region.edge_eta_division()
    elif config['division'] == 'none':
        div = None
    else:
        raise ValueError('Division must be spcified as \'new_phi\' \'z\',\'phi\', \'eta\' or \'layer\'.')

    if div is not None:
        region_ids = div.get_region_ids()
        if config['region_id'] not in region_ids:
            raise ValueError('Please specify a valid region_id. Choose from:', region_ids,'. You passed:', config['region_id'])
    
    sample_flag = config['sample_flag'] #whether to load all data or just a sample (to dev, debug)
    #load train data - needed for quantum-enhanced even if model already trained
    train_data = load_events.load_events(config['num_train'], config['tracklet_dataset'], 'train', sample_flag)
    if div is not None:
        train_data_in_region = train_data[train_data['object_coords'].apply(div.find_region) == config['region_id'] ].reset_index(drop = True)
    else:
        train_data_in_region = train_data
    
    #transform data
    data_transform(train_data_in_region, method = config['data_scaler'])

    #(add physics)
    add_physics_to_tracklet(train_data_in_region, config['add_to_tracklet'])

    train_tracklets_in_region = [np.array(train_data_in_region['object'].values[item]) for item in range(len(train_data_in_region))]
    train_labels_in_region = [np.array(train_data_in_region['label'].values[item]) for item in range(len(train_data_in_region))]

    #save labels from the training data, used for computing complexity of kernel
    tr_labels_dir = str(Path().absolute() / 'saved_tr_labels')
    tr_labels_filename = 'train_labels_'+config['tracklet_dataset']+'_tr_'+str(config['num_train'])+'_reg_'+str(config['region_id'])+'_in_'+config['division']+'.npy'
    np.save(tr_labels_dir+'/'+tr_labels_filename, train_labels_in_region)

    tracklet_dimension = np.shape(train_tracklets_in_region)[1]
    if not (config['circuit_width'] == tracklet_dimension):
        raise AssertionError('Tracklet object has dimension: ', tracklet_dimension, '.',
        ' Feature dimension expected: ', config['circuit_width'],'.')
    
    if config['hyperparam_opt_flag']:
        #train-validation split obtained from train data for hyperparam optimisation
        #no test data loaded
        #NOTE: this is heavily based on Mohammad's code
        
        kfold = KFold(n_splits = 5, shuffle = True, random_state=1)
        #this has shape (5,2) where each of the 5 tuples contains two lists of indeces: ([train, test])
        indices = list(kfold.split(train_tracklets_in_region))
        train_id, valid_id = indices[config['fold_id']]
        print(f'valid_id: {valid_id}')
        print(np.shape(train_tracklets_in_region))
        #note the naming of these isn't the most instructive (test instead of valid, re-defining train)
        ##but I'm trying to be consistent with the rest of the code
        test_tracklets_in_region = [train_tracklets_in_region[i] for i in valid_id]
        test_labels_in_region = [train_labels_in_region[i] for i in valid_id]

        train_labels_in_region = [train_labels_in_region[i] for i in train_id]
        train_tracklets_in_region = [train_tracklets_in_region[i] for i in train_id]

        
        

    t_start_train = time.time()
    print(f'Time elapsed before training starts: {t_start_train - t_start} s.')

    if config['load_model'] == False:
        #train model
        QKE_model.set_model(load = False, 
        train_data = train_tracklets_in_region, 
        train_labels = train_labels_in_region, 
        from_config = config_filename)
    else:
        #load_model
        model = joblib.load(config['model_file'])
        QKE_model.set_model(load = True, model = model)

    t_end_train = time.time()
    print(f'Time taken to train: {t_end_train - t_start_train} s.')

    if not config['hyperparam_opt_flag']:
        #load test data
        test_data = load_events.load_events(config['num_test'], config['tracklet_dataset'], 'test', sample_flag)
        if div is not None:
                test_data_in_region = test_data[test_data['object_coords'].apply(div.find_region) == config['region_id']].reset_index(drop = True)
        else:
            test_data_in_region = test_data
    
        #transform data
        data_transform(test_data_in_region, method = config['data_scaler'])

        #(add physics)
        add_physics_to_tracklet(test_data_in_region, config['add_to_tracklet'])

        test_tracklets_in_region = [np.array(test_data_in_region['object'].values[item]) for item in range(len(test_data_in_region))]

    #test
    if config['keep_kernel']:
        model_predictions, decision_functions = QKE_model.test(test_tracklets_in_region, train_tracklets_in_region)
    else:
        model_predictions, decision_functions = QKE_model.test(test_tracklets_in_region)

    t_end_test = time.time()
    print(f'Time taken to test: {t_end_test - t_end_train} s.')

    if config['hyperparam_opt_flag']:
        #no need to save full dataframes when doing hyperparam opt
        #save only timing and score information
        #NOTE: when running on the cluster, print statements are saved in output files (.o)
        ##and info can be retrieved for further analysis
        print(f'roc score: {roc_auc_score(test_labels_in_region, decision_functions)}')
        print(f'f1 score: {f1_score(test_labels_in_region, model_predictions)}')
        print(f'accuracy score: {accuracy_score(test_labels_in_region, model_predictions)}')
        print(f'precision score: {precision_score(test_labels_in_region, model_predictions)}')
        print(f'recall score: {recall_score(test_labels_in_region, model_predictions)}')
    else:
        #update dataframe with prediction column
        test_data_in_region.insert(np.shape(test_data_in_region)[1], 'prediction', model_predictions)
        test_data_in_region.insert(np.shape(test_data_in_region)[1], 'decision_function', decision_functions)
        #save resulting dataframe
        results = test_data_in_region.to_numpy()
        tracklet_type = config['tracklet_dataset']
        if isinstance(config['add_to_tracklet'], list):
            tracklet_type = 'with_physics_'+tracklet_type
        #add kernel info to saved file
        tracklet_type = config['kernel_type']+'_'+tracklet_type
        results_file = tracklet_type+'_predictions_'+str(config['num_train'])+'_'+str(config['num_test'])+'_events_reg_'+str(config['region_id'])+'_in_'+str(config['division'])

        results_path = str(Path().absolute() /'predictions')+'/'+results_file #(!)
        np.save(results_path, results)

    t_end = time.time()
    print(f'Running the code took: {t_end - t_start} s.')

    #prof.disable() #(!)
    # with open(profiling_stats_path, 'w') as stream: #(!)
    #    stats = pstats.Stats(prof, stream = stream).strip_dirs().sort_stats('cumtime') #(!)
    #    stats.print_stats(20) #(!)