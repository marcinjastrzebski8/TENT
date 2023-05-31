import yaml
import unittest
from pathlib import Path

def produce_config(config, file_name):
    """
    Makes a .yaml file with a given name
    """
    file_path = 'configs/auto_config_'+file_name+'.yaml'
    with open(file_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style = False)
    print('Config file was produced at:')
    print(file_path)
    return file_path

def set_config_name(config, hyperparam_opt = False):
    """
    config: a config dictionary (not a .yaml file)
    """
    filename = config['tracklet_dataset']+'_'+config['kernel_type']
    if 'classical' in config['kernel_type']:
        filename += '_C_'+str(config['C_class']).replace('.','p') #remove dots from names [coming from floats]
        filename += '_gamma_'+str(config['gamma_class']).replace('.','p')
    else:
        filename += '_alpha_'+str(config['alpha']).replace('.','p')
        filename += '_C_'+str(config['C_quant']).replace('.','p')
        filename += '_'+str(config['paulis'][0]) #one_qubit_rotation
        filename += '_'+str(config['paulis'][1]) #two_qubit_rotation

    filename += '_div_'+config['division']
    
    if hyperparam_opt:
        #NOTE: this could be integrated into all names but the format would clash with old results
        filename += '_ent_'+config['entanglement']
        filename +='_fold_'+str(config['fold_id'])
    else:
        filename += '_reg_'+config['region_id']
        filename += '_tr_'+str(config['num_train'])
        filename += '_te_'+str(config['num_test'])
    return filename

def produce_shell_script(config_path, config_name: str):
    """
    Create a .sh file which can be run to send multiple jobs to a batch farm.
    """
    file = 'shell_jobs/job_'+config_name.removesuffix('.yaml')+'.sh'
    with open(file, 'w') as f:
        #I don't know if I should add a memory request
        f.write('#usr/bin/bash\n')
        f.write('\n')
        f.write('#PBS -N ' + config_name+'\n')
        f.write('\n')
        f.write('#PBS -k o\n')
        f.write('\n')
        f.write('#PBS -j oe\n')
        f.write('\n')
        f.write('#PBS -q long\n')
        f.write('\n')
        f.write('module load dot\n')
        f.write('source activate qiskit_env\n')
        f.write('\n')
        f.write('(cd /unix/atlastracking/marcinj/TENT/TENT && python3 classify_tracklets.py '+config_path+')\n')

def load_config(file_path = None):
    path = Path(file_path)
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader = yaml.FullLoader)
    #returns the model and the name of the config run
    return cfg, path.stem

#Unit testing
class TestLoad(unittest.TestCase):
    """
    Test config files can be read appropriately on an example config file.
    Note: to test, need to run from top of TENT project
    """
    def test_loader(self):
        path_to_example = Path().absolute() / 'TENT/sliced_detector_analysis/configs/example.yaml'
        test_variables, filename = load_config(path_to_example)
        print(test_variables)
        self.assertEqual(test_variables['division'], 'new_phi')
        self.assertEqual(filename, 'example')

if __name__ == '__main__':
    unittest.main()
