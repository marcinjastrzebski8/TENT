"""
Updating the pipeline to be readible and general for any objects used for classification.
"""

import unittest
import os
import numpy as np
import pandas as pd
path_for_imports = os.path.abspath('.')
def initial_data_to_dataframe(folder_str, num_events) -> pd.DataFrame:
    """
    NOTE: This is essentially depricated, keeping just in case.
    Tuysuz-processed edge data came in a format different to the one I want to use.
    Converts it into a pandas DataFrame

    folder_str: str, file_folder_str is passed in the main function
    num_events: int, event_files from main function
    """
    files_folder_as_list = os.listdir(folder_str)
    files_folder_as_list.sort()

    data_files = files_folder_as_list[:num_events] #this is a list of strings
    data_list = [np.load(folder_str+'/'+event) for event in data_files] #this is a list of numpy arrays
    data = np.concatenate(data_list) #list of edges
    
    #only way I could find to make a dataframe with a list (edge) as a single row entry
    df_exploded = pd.DataFrame(data, columns = [str(i) for i in range(6)])
    df_exploded['edge'] = df_exploded.values.tolist()
    df_combined = df_exploded[['edge']]

    labels_files = files_folder_as_list[50:50+num_events] 
    labels_list = [np.load(folder_str+'/'+event) for event in labels_files] 
    labels = np.concatenate(labels_list)

    df_combined = df_exploded[['edge']]
    df_combined['label'] = labels

    return df_combined

def load_events(event_files, dataset = 'edge', folder = None, sample = False) -> pd.DataFrame:
    """
    Fetches train and test data along with their labels.
    Can specify which database to work with.

    event_files: int in range <1;20>, (test) or <1,80> (train) how many events to test/train on 
    database:    'initial' is Tuysuz pre-processed edges (redundant soon), 'edge', 'triplet', 'io_triplet_pair', 'inner_triplet'
    folder:      'train' or 'test'
    sample:      if True, only read a subset of tracklets from a given event. 
    """

    #make sure this path points to the right folders
    database_folder_str = path_for_imports+'/data'
    files_folder_str = database_folder_str+'/'+dataset+'_data/'+folder
    if dataset == 'initial':
        events_data = initial_data_to_dataframe(files_folder_str, event_files)
    elif dataset in ('edge', 'triplet', 'balanced_triplet', 'io_triplet_pair', 'inner_triplet', 'quintuplet'):
        if dataset == 'edge':
            column_names = ['object_coords', 'label', 'eta', 'phi', 
            'true_pt', 'layer','track_length', 'hit1_id', 'hit2_id', 'particle_num']
        elif dataset in ('triplet', 'balanced_triplet', 'inner_triplet'):
            column_names = ['object_coords','label', 'phi_breaking','theta_breaking','pt_estimate', 'ip','pt','layers', 'track_lengths',
            'particle_id', 'particle_num', 'phi', 'eta']
        elif dataset == 'io_triplet_pair':
            column_names = ['object_coords', 'label', 'phi_breaking', 'theta_breaking', 'pt_estimate', 'ip', 'pt', 'track_length', 'particle_id', 'particle_num', 'phi', 'eta']
        elif dataset == 'quintuplet':
            column_names = ['object_coords','label', 'phi_breaking','theta_breaking','pt_estimate', 'ip','pt','track_lengths',
            'particle_id', 'particle_num', 'phi', 'eta', 'seed_prediction']
        files_folder_as_list = os.listdir(files_folder_str)
        files_folder_as_list.sort()
        data_files = files_folder_as_list[:event_files] #this is a list of strings
        if sample:
            data_list = [pd.DataFrame(np.load(files_folder_str+'/'+event, allow_pickle = True),
            columns = column_names)[:200] for event in data_files]
        else:
            data_list = [pd.DataFrame(np.load(files_folder_str+'/'+event, allow_pickle = True),
            columns = column_names) for event in data_files]
        events_data = pd.concat(data_list).reset_index(drop = True)
    else:
        raise ValueError('database must be chosen from (\'initial\', \'edge\', \'triplet\', \'io_triplet_pair\')')
    return events_data

class TestLoad(unittest.TestCase):
    """
    Test load_events returns a single DataFrame and works for two possible databases
    """
    def test_load(self):
        tuy_data = load_events(10, dataset = 'initial',folder = 'test')
        edge_data = load_events(40, dataset = 'edge', folder = 'train')
        triplet_data = load_events(40,dataset = 'triplet', folder = 'train')
        print(triplet_data)
        self.assertEqual(np.shape(tuy_data)[1], 2)
        #THIS ADJUSTED BECAUSE OF UPDATE - SHAPE USED TO BE 9
        self.assertEqual(np.shape(edge_data)[1], 8)
        self.assertEqual(np.shape(triplet_data)[1], 13)

if __name__ == '__main__':
    unittest.main()
