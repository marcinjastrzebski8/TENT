"""
Tools used to perform a hyperparam optimisation (combinatorial)
main script is h_o_submit_several_configs.py
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
from pathlib import Path
import os
import re
import unittest

from config_tools import load_config

def extract_results_info(results_file):
    with open(results_file, 'r') as file:
        h_o_run_info = {'hyper_params':'',
        'time_tr': '',
        'time_te': '',
        'roc': '',
        'f1': '',
        'acc': '',
        'prec': '',
        'rec':''}
        
        #read output file of a training run
        lines = file.readlines()
        if not lines:
            print(f'FILE IS EMPTY: {results_file}')
        else:
            for line in lines:
                #NOTE: regex is strange
                if line.startswith('SVC model trained and stored'):
                    #this not added to the dict yet, needs processing
                    model_name = re.search(r'SVC model trained and stored as:\s+(.*)', line).group(1)
                if line.startswith('Time taken to train'):
                    h_o_run_info['time_tr'] = float(re.search(r'Time taken to train:\s+(\d+\.+\d+)', line).group(1))
                if line.startswith('Time taken to test'):
                    h_o_run_info['time_te'] = float(re.search(r'Time taken to test:\s+(\d+\.+\d+)', line).group(1))
                if line.startswith('roc score'):
                    h_o_run_info['roc'] = float(re.search(r'roc score:\s+(\d+\.+\d+)', line).group(1))
                if line.startswith('f1 score'):
                    h_o_run_info['f1'] = float(re.search(r'f1 score:\s+(\d+\.+\d+)', line).group(1))
                if line.startswith('accuracy score'):
                    h_o_run_info['acc'] = float(re.search(r'accuracy score:\s+(\d+\.+\d+)', line).group(1))
                if line.startswith('precision score'):
                    h_o_run_info['prec'] = float(re.search(r'precision score:\s+(\d+\.+\d+)', line).group(1))
                if line.startswith('recall score'):
                    h_o_run_info['rec'] = float(re.search(r'recall score:\s+(\d+\.+\d+)', line).group(1))
            if any([item == '' for item in list(h_o_run_info.values())[1:]]):
                print('SOME INFORMATION IS MISSING FROM THE RUN')
        #extract information about the hyperparameters from appropriate line
        if lines:
            model_elements = model_name.split('_')
            model_params = {}
            if not any(model_elements) == 'classical':
                for i, element in enumerate(model_elements):
                    #catch which type of kernel
                    if element == 'fidelity':
                        model_params['kernel_type'] = element
                    if element == 'projected_1rdm':
                        model_params['kernel_type'] = element

                    if element == 'alpha':
                        alpha = model_elements[i+1]
                        if 'p' in alpha:
                            alpha = alpha.replace('p', '.') #p corresponds to a dot in a float
                        model_params['alpha'] = float(alpha)
                    if element in ('X', 'Y', 'Z'):
                        one_qubit_gate = element
                        two_qubit_gate = model_elements[i+1]
                        paulis = one_qubit_gate+'_'+two_qubit_gate
                        model_params['paulis'] = paulis
                    if element == 'ent':
                        ent = model_elements[i+1]
                        model_params['ent'] = ent
            else:
                for i, element in enumerate(model_elements):
                #TODO: ADD THIS WHEN WORKING WITH CLASSICAL KERNELS
                    model_params['kernel_type'] = 'classical'
                    pass
            for i, element in enumerate(model_elements):
                if element == 'C':
                    C = model_elements[i+1]
                    if 'p' in C:
                        C = C.replace('p', '.') #p corresponds to a dot in a float
                    model_params['C'] = float(C)
                if element == 'fold':
                    fold = model_elements[i+1].replace('.sav', '')
                    model_params['fold'] = int(fold)
            h_o_run_info['hyper_params'] = model_params
    return h_o_run_info

def make_run_dataframe(kernel_type: str, results_folder: str):
    #NOTE: this is heavily based on Mohammads addResultToFile.py from galaxy morphology
    """
    TODO: decide on what the read file will be named
    """
    #make the columns of the dataframe
    cols = []
    metrics = ['time_tr', 'time_te', 'roc', 'prec', 'acc', 'f1', 'rec']
    for metric in metrics:
        for fold_id in range(5):
            column = metric+f'_f{fold_id}'
            cols.append(column)
        avg_col = f'avg_{metric}'
        std_col = f'std_{metric}'
        cols.append(avg_col)
        cols.append(std_col)

    if kernel_type in ['fidelity', 'projected_1rdm']:
        cols.append('alpha')
        cols.append('ent')
        cols.append('paulis')
        if kernel_type == 'projected_1rdm':
            cols.append('gamma_quant')
    elif kernel_type == 'classical':
        cols.append('gamma')
    else:
        raise ValueError(f'wrong kernel type passed: {kernel_type}')
    cols.append('C')
    df = pd.DataFrame(columns = cols)

    folder_name = f'hyperparam_opt/h_o_results/{results_folder}'
    results_path = Path().absolute() / folder_name

    for results_file in os.listdir(results_path):
        info = extract_results_info(str(results_path)+'/'+results_file)
        hyper_params = info['hyper_params']
        fold = hyper_params.pop('fold')
        df_index_hyper_params = tuple(hyper_params.values())

        #each hyperparam model gets its own row which should contain info from 5 folds
        new_df_point = pd.DataFrame({
            f'time_tr_f{fold}': info['time_tr'],
            f'time_te_f{fold}': info['time_te'],
            f'roc_f{fold}': info['roc'],
            f'f1_f{fold}': info['f1'],
            f'acc_f{fold}': info['acc'],
            f'prec_f{fold}': info['prec'],
            f'rec_f{fold}': info['rec']
        }, index = [df_index_hyper_params])

        if df_index_hyper_params in df.index:
            df.update(new_df_point)
        else:
            df = pd.concat([df, new_df_point], ignore_index = False)
    #compute averages from collected folds
    for metric in metrics:
        df[f'avg_{metric}'] = df.loc[:,f'{metric}_f{0}': f'{metric}_f{4}'].dropna(axis = 1).mean(axis = 1)
        df[f'std_{metric}'] = df.loc[:,f'{metric}_f{0}': f'{metric}_f{4}'].dropna(axis = 1).std(axis = 1)
    return df



def load_kernel(config_name):
    """
    Given a config file, find the corresponding kernel which has been produced.
    TODO: dev
    """
    local_path = 'configs/auto_config_'+config_name+'.yaml'
    config_path = str(Path().absolute() / local_path)
    cfg, cfg_stem = load_config(config_path)
    kernel_name = 'kernel_'+cfg_stem+'.npy'
    kernel_file = str(Path().absolute() / 'saved_kernels' / kernel_name)
    return np.load(kernel_file)

def calculate_approximate_dimension(k):
    """
    NOTE: this code copied from the QuASK project
    Calculate the approximate dimension (d), which is equation S111 in the Supplementary
    of the "The power of data in quantum machine learning"
    (https://www.nature.com/articles/s41467-021-22539-9).

    Args:
        k: Kernel gram matrix.

    Returns:
        approximate dimension of the given kernel (float).
    """
    u, t, udagger = la.svd(k, full_matrices=True)

    N = len(t)

    d = 0
    for i in range(N):
        d += 1 / (N - i) * sum(t[i:])
    return d

def check_eff_dim(config_name):
    """
    Compute effective dimension of a kernel from an output of quantum kernel training.
    TOOD: dev 
    """
    kernel = load_kernel(config_name)
    return calculate_approximate_dimension(kernel)

#make_run_dataframe('quantum')

class TestResultsReading(unittest.TestCase):
    
    def test_extract_params(self):
        #NOTE: this only checks for a case of a (specific, example) quantum kernel
        unittest_result_file = '/Users/marcinjastrzebski/Desktop/ACADEMIA/SECOND_YEAR/TENT/hyperparam_opt/h_o_results/unittest_results/results_for_unittest.o6324786'
        info = extract_results_info(unittest_result_file)
        hyper_params = info['hyper_params']
        self.assertAlmostEqual(info['time_tr'], 0.88, 2)
        self.assertAlmostEqual(info['time_te'], 0.75, 2)
        self.assertAlmostEqual(info['roc'], 0.375, 2)
        self.assertAlmostEqual(info['f1'], 0.8, 2)
        self.assertAlmostEqual(info['acc'], 0.666, 2)
        self.assertAlmostEqual(info['prec'], 0.666, 2)
        self.assertAlmostEqual(info['rec'], 1.0, 2)
        
        self.assertAlmostEqual(hyper_params['alpha'], 1, 2)
        self.assertEqual(hyper_params['paulis'], 'Z_ZZ')
        self.assertEqual(hyper_params['ent'],'linear')
        self.assertAlmostEqual(hyper_params['C'], 1.0, 2)
        self.assertEqual(hyper_params['fold'],2,)
        self.assertEqual(hyper_params['kernel_type'], 'fidelity')

if __name__ == '__main__':
    unittest.main()