"""
This is submit_several_configs with different options for the loop
Repurposed for hyperparam optimisation
"""
#NOTE: LOTS TO DO HERE, FOCUS N CARRY ON
import numpy as np
import subprocess
import sys
from config_tools import produce_config, set_config_name, produce_shell_script
from hyperparam_opt.h_opt_tools import check_eff_dim

def run(config_path, config_name, submit_to_batch):
    """
    Run an SVM job as specified by a config file;
    Can be done locally (mostly for developing and checking)
    Or on the cluster (where jobs can be parallelised)
    """
    max_runtime = 200 #in seconds, applies when running locally (not submitted to batch)
    if submit_to_batch:
        subprocess.run(['qsub', 'shell_jobs/job_'+config_name+'.sh'])
        return 0
    else:
        try:
            #run locally, if takes too long - will need to be submitted
            command = 'python3 classify_tracklets.py'.split()
            command.append(config_path)
            subprocess.run(command, timeout=max_runtime)
            return 0
        except subprocess.TimeoutExpired:
            print('TOOK TOO LONG. TERMINATED. RUN ON BATCH (yourself).')
            print('************************************************')
            with open('long_jobs.txt', 'a') as file:
                file.write(config_path+"\n")
            return 1
    
def produce_job(config, submit_to_batch: bool, hyperparam_opt:bool):
    config_name = set_config_name(config, hyperparam_opt)
    config_path = produce_config(config, config_name)
    if submit_to_batch:  
        produce_shell_script(config_path, config_name)
        return run(config_path, config_name, submit_to_batch)
    else:
        return run(config_path, config_name, submit_to_batch)

def pass_threshold(list_of_dims, threshold) -> bool:
    return([entry > threshold for entry in list_of_dims])


def main(batch_job, project_kernels):
    #note (!) means filled inside the function and iterated over
    config = dict(
        load_model = False,
        tracklet_dataset = 'inner_triplet', #TODO: change to quintuplet
        division = 'new_phi', 
        region_id = '0',
        data_scaler = 'inner_global', #(!) #TODO: change to quint_global when that's a valid option 
        add_to_tracklet = None,
        num_test = 0, #this should be irrelevant for hyperparam opt
        num_train = 1, #(!!!) TODO: decide on number of events
        class_weight = {0: 1.0, 1: 1.0},
        kernel_type = None, #Quantum in this script; start with fidelity and project if dimension is large
        gamma_class = None, #irrelevant here 
        gamma_quant = 1, #this could in principle also be optimised
        C_class = 0, #irrelevant here
        C_quant = None, #(!)
        alpha = None, #(!)
        paulis = None, #(!)
        entanglement = None, #(!)
        circuit_width = 9, #TODO: change to 15 when got quintuplet data
        keep_kernel = True,
        sample_flag = False, #MAKE SURE THIS IS OFF ON CLUSTER
        hyperparam_opt_flag = True,
        fold_id = None #(!) this is ignored if hyperparam_opt_flag == False
    )
    #hyperparams for the scan 
    one_qubit_paulis = ['X', 'Y', 'Z']
    two_qubit_paulis = ['XX', 'YY', 'ZZ'] #should add XY YX ZY YZ ZX XZ :/
    entanglements = ['full', 'linear', 'circular']
    alphas = [0.05, 0.1, 1, 5]
    Cs = np.logspace(-3,3,7)
    
    fold_ids = range(5)
    for one_q_p in one_qubit_paulis:
        for two_q_p in two_qubit_paulis:
            config['paulis'] = [one_q_p, two_q_p]
            for alpha in alphas:
                config['alpha'] = alpha
                for C in Cs:
                    config['C_quant'] = float(C)
                    for entanglement in entanglements:
                        config['entanglement'] = entanglement

                        #this will be changed if projected_kernels flag is True
                        config['kernel_type'] = 'fidelity'

                        if project_kernels:
                            #NOTE: this can only be done after the fidelity equivalent search has been done

                            
                            #check if any of the fidelity folds passed the threshold
                            threshold = (2**config['circuit_width'])/2
                            fold_kernel_dims = []

                            for fold_id in fold_ids:
                                config['fold_id'] = fold_id
                                fidelity_config_name = set_config_name(config, hyperparam_opt=True)
                                kernel_dim = check_eff_dim(fidelity_config_name)
                                fold_kernel_dims.append(kernel_dim)
                            large_dim_found = any(pass_threshold(fold_kernel_dims, threshold))

                            if large_dim_found:
                                print('Some of the folds\' kernels had dimension larger than the threshold')
                                print(f'Threshold: {threshold}')
                                print(f'Fold kernel dimensions: {fold_kernel_dims}')
                                print('Proceeding to produce projected kernels')
                                config['kernel_type'] = 'projected_1rdm'
                            else:
                                print('All folds produced kernels passing the threshold')
                                print(f'Threshold: {threshold}')
                                print(f'Fold kernel dimensions: {fold_kernel_dims}')
                                break
                            
                        run_success = 0 #not relevant for batch jobs

                        #Hyperparams set, now run five folds for that setting
                        for fold_id in fold_ids:
                            print(f'On fold {fold_id}')
                            config['fold_id'] = fold_id

                            #if one fold fails locally, break out of all next folds
                            if run_success !=0:
                                print('gon break bro')
                                break
                            run_success = produce_job(config, batch_job, config['hyperparam_opt_flag'])
                                
                    
if __name__ == '__main__':
    batch_flag = False #could have this True on the cluster and False locally?
    main(batch_flag, True)