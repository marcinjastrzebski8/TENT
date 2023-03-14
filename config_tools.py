import yaml

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

def set_config_name(config):
    """
    Need to think about all the information that can identify a job
    config: a config dictionary (not a .yaml file)
    """
    filename = config['kernel_type']
    if 'classical' in config['kernel_type']:
        filename += '_C_'+str(config['C_class']).replace('.','p') #remove dots from names [coming from floats]
        filename += '_gamma_'+str(config['gamma_class']).replace('.','p')
        #what else
    else:
        filename += '_alpha_'+str(config['alpha']).replace('.','p')
        filename += '_C_'+str(config['C_quant']).replace('.','p')
        filename += '_'+str(config['paulis'][0]) #one_qubit_rotation
        filename += '_'+str(config['paulis'][1]) #two_qubit_rotation

    filename += '_div_'+config['division']
    filename += '_reg_'+config['region_id']
    filename += '_tr_'+str(config['num_train'])
    filename += '_te_'+str(config['num_test'])
    return filename
