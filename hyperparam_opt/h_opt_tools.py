"""
Tools used to perform a hyperparam optimisation (combinatorial)
main script is h_o_submit_several_configs.py
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
from pathlib import Path

from config_tools import load_config



def make_run_dataframe(h_opt_run_specs):
    """
    TODO: decide on what the read file will be named
    """
    df_cols_times = [f'f{i}_time' for i in range(5)]
    df_cols_aucs = [f'f{i}_auc' for i in range(5)]
    df_cols = ['f0_time']
    df = pd.DataFrame(columns=df_cols)


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