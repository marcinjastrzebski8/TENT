import subprocess
import sys
from config_tools import produce_config, set_config_name

def run(config, submit_to_batch):
    file_name = set_config_name(config)
    file_path = produce_config(config, file_name)
    max_runtime = 200 #in seconds, applies when running locally (not submitted to batch)
    if submit_to_batch:
        #need to figure out how many .sh files I need and what names they should have
        subprocess.run(['qsub', '-v', 'input='+file_path, 'WHAT_SHOULD_THE_NAME_BE.sh'])
    else:
        try:
            #run locally, if takes too long - will need to be submitted
            command = 'python3 classify_tracklets.py'.split()
            command.append(file_path)
            subprocess.run(command, timeout=max_runtime)
        except subprocess.TimeoutExpired:
            print('TOOK TOO LONG. TERMINATED. RUN ON BATCH (yourself).')
            print('************************************************')
            with open('long_jobs.txt', 'a') as file:
                file.write(file_name+"\n")
            return 1
        return 0

def main(submit_to_batch):
    #note (!) means filled inside the code
    config = dict(
        load_model = False,
        tracklet_dataset = None, #(!)
        division = 'new_phi', 
        region_id = None, #(!!) This is a special variable; if run on the batch, it's submitted as 16 jobs
        data_scaler = None, #(!)
        add_to_tracklet = None,
        num_test = None, #(!) 
        num_train = None, #(!)
        class_weight = {0: 1.0, 1: 1.0},
        kernel_type = None, #(!)
        gamma_class = 1, 
        gamma_quant = 1,
        C_class = 1.0e+6,
        C_quant = 1.0e+6,
        alpha = 0.2,
        paulis = ['Z', 'YY'],
        circuit_width = 9,
        model_saved_path = 'trained_models/',
        result_output_path = 'output/',
        keep_kernel = True,
        sample_flag = True
    )
    #what to iterate over, essentially defines what analysis can be done
    #i can and should defo parallelise (-t $PBS_ARRAYID) the different regions [==models] so don't include here
    #note: but that means I'll need to call the config files accordingly
    train_size_list = [[1,2,3], [5,10,15]]
    test_size_list = [[1],[5]]
    tracklet_type_list = ['triplet', 'inner_triplet']
    data_scaler_list = ['global', 'inner_global']
    kernel_type_list = ['classical_keep_kernel', 'fidelity', 'projected_1rdm'] #class, quant, quant-projected [... to come?]
  
    for i, tracklet_type in enumerate(tracklet_type_list):
        config['data_scaler'] = data_scaler_list[i] #depends on tracklet type
        config['tracklet_dataset'] = tracklet_type
        for kernel_type in kernel_type_list:
            config['kernel_type'] = kernel_type
            #loops below are going to fail if the innermost loop fails, use this to determine whether to break out of them
            run_success = 0 
            for r_id in [str(i) for i in range(16)]:
                config['region_id'] = r_id
                if run_success !=0:
                    break
                for te_size in test_size_list[i]:
                    config['num_test'] = te_size
                    if run_success !=0:
                        break
                    for tr_size in train_size_list[i]:
                        config['num_train'] = tr_size
                        run_success = run(config, submit_to_batch)
                        #if jobs run long for a given train size then next (train sizes, test sizes, regions) will take >= time 
                        if run_success !=0:
                            print('gon break bro')
                            break

if __name__ == '__main__':
    batch_flag = False #could have this True on the cluster and False locally?
    main(batch_flag)