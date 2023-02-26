"""
Randomly assigns objects to be true according to the expected fraction of true segemnts in the pre-processed data.
"""
import unittest
import numpy as np
import pandas as pd
import random

def random_classification(data_set, expected_purity) -> pd.DataFrame:
    """
    Randomly classify objects in a pre-processed data set based on expected purity. 
    To be used as benchmark for SVM.

    data_set:        DataFrame with the data to be classified
    expected_purity: float, expected fraction of true objects in the data
    """
    data_size = np.shape(data_set)[0]
    true_generation_weight = expected_purity/(1 - expected_purity)
    #list of random 1s and 0s
    random_predictions = random.choices(population = [0,1],weights = [1, true_generation_weight],k = data_size) 
    data_set['prediction'] = random_predictions
    #this is only for consistency in loading the data, add a blank column
    blank_column = pd.Series(dtype = object)
    data_set.insert(0, 'object', blank_column)

    return data_set

#TESTS - should gather somewhere once I've more of them?
class TestNewDf(unittest.TestCase):
    """
    Test random_classification adds predictions column with appropriate number of True predictions to input DataFrame
    """
    def test_new_df(self):
        fake_data_size = 100000
        fake_data = np.ones([fake_data_size,2])
        fake_data_set = pd.DataFrame(fake_data, columns = ['data', 'labels'])
        new_df = random_classification(fake_data_set, 0.82)
        true_predictions = new_df['prediction'][new_df['prediction'] == 1]
        how_many_trues = np.shape(true_predictions)[0]

        self.assertEqual(np.shape(new_df)[1], np.shape(fake_data)[1] + 2)
        self.assertEqual(new_df.columns[-1], 'prediction')
        self.assertAlmostEqual(how_many_trues/fake_data_size, 0.82, places = 2)

if __name__ == '__main__':
    unittest.main()
