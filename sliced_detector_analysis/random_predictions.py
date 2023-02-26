import argparse
import os
import numpy as np
import pandas as pd
import sys

from tools import (find_detector_region as find_region, random_classification, load_events)

def classify_edges(num_events, database = 'edges', division = '0', region_id = '0', purity = 0.5):
    """
    Imitate prediction pipeline but make random guesses as predictions.
    """
    #choose detector division 
    squished_division = False
    if database == 'initial':
        squished_division = True #at the moment this doesn't work, can't use Tuysuz's data
    #choose detector division 
    if division == 'new_phi':
        div = find_region.edge_new_phi_division()
    elif division == 'z':
        div = find_region.edge_z_division()
    elif division == 'layer':
        div = find_region.edge_phi_division()
    elif division == 'phi':
        div = find_region.edge_phi_division()
    elif division == 'eta':
        div = find_region.edge_eta_division()
    else:
        raise ValueError('Division must be spcified as \'new_phi\' \'z\',\'phi\', \'eta\' or \'layer\'.')

    region_ids = div.get_region_ids()
    if region_id not in region_ids:
        raise ValueError('Please specify a valid region_id. Choose from:', region_ids,'. You passed:',region_id)
   
    test_data = load_events.load_events(num_events, database, 'test')
    #divide edges accordingly
    test_data_in_region = test_data[test_data['object_coords'].apply(div.find_region) == region_id].reset_index(drop = True)

    #at the moment some defined regions are empty - mistake in division definition
    if  len(test_data_in_region) == 0:
        raise ValueError("The specified region is empty. Try a different one.")

    #add prediction column to data 
    random_classification.random_classification(test_data_in_region, purity)
    results = test_data_in_region.to_numpy()
    results_file = 'random_'+database+'_predictions_'+str(num_events)+'_'+str(num_events)+'_events_reg_'+str(region_id)+'_in_'+str(division)
    np.save(results_file, results) #load with allow_pickle = True
    
parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('num_files',type = int)
add_arg('database')
add_arg('division')
add_arg('region_id')
add_arg('purity', type = float)
args = parser.parse_args()

parse_num_files = args.num_files
parse_database = args.database
parse_division = args.division
parse_region = args.region_id
parse_purity = args.purity

classify_edges(num_events = parse_num_files, database = parse_database, division = parse_division, region_id = parse_region, purity = parse_purity)