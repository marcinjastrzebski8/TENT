"""
Use data provided to train and save an svm classifier
Can use a classicla svm or quantum-enhanced
STATUS: in dev, job report could be compiled in main
"""

import numpy as np
import sys
import os

from sklearn.svm import SVC
import joblib

from qiskit import (Aer,IBMQ)
IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(group='open')
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel

path_for_imports = os.path.abspath('.')
print(path_for_imports)
sys.path.append(path_for_imports) #for quantum_circuit
from quantum_circuit import (param_feature_map, param_U_gate)

class QKE_SVC():
    """
    Defines an SVC classifier model family - either classical or quantum-enhanced.
    Can be used to train/load a given instance of the family.
    Classical case is an rbf.
    Quantum is Havlicek-based (IPQ).
    """
    def __init__(self,
    classical,
    class_weight, 
    gamma = None, 
    C_class = None, 
    alpha = None,
    C_quant = None,
    single_mapping = None,
    pair_mapping = None,
    interaction = None,
    circuit_width = None):
        if classical:
            self.gamma = gamma
            self.C_class = C_class
        else:
            self.alpha = alpha
            self.C_quant = C_quant
            self.single_mapping = single_mapping
            self.pair_mapping = pair_mapping
            self.interaction = interaction

            self.backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
            self.circuit_width = circuit_width
            params = ParameterVector('phi', circuit_width)
            U_gate = param_U_gate.U_flexible(circuit_width, 
            params, 
            single_mapping = self.single_mapping, 
            pair_mapping = self.pair_mapping, 
            interaction = self.interaction, 
            alpha = self.alpha) 
            tracklet_feature_map = param_feature_map.feature_map(circuit_width, U_gate)

            self.kernel = QuantumKernel(feature_map = tracklet_feature_map, quantum_instance = self.backend)
        self.classical = classical
        self.class_weight = class_weight
        self.cache_chosen = 1000

    def train_model(self, train_data, train_labels, from_config):
        """
        When using a precomputed kernel fit and predict differ.
        Thus classical an quantum implementations differ.
        """
        if self.classical:
            model = SVC(kernel = 'rbf', 
            gamma = self.gamma,
            C = self.C_class,
            cache_size = self.cache_chosen,
            class_weight = self.class_weight)

            model.fit(train_data, train_labels)
        else:
            model = SVC(kernel = 'precomputed',
            C = self.C_quant,
            cache_size = self.cache_chosen,
            class_weight = self.class_weight)
    
            train_matrix = self.kernel.evaluate(x_vec = train_data)
            model.fit(train_matrix, train_labels)
        #save fitted SVC model
        filename = 'model_from_'+from_config+'.sav'
        joblib.dump(model, filename)
        print('SVC model trained and stored as:', filename)
        return model

    def set_model(self, load, model = None, train_data = None, train_labels = None, from_config = None):
        if load:
            self.model = model
            print('model has been loaded, model: ', self.model)
        else:
            self.model = self.train_model(train_data = train_data, train_labels = train_labels, from_config = from_config)

    def test(self, test_data, train_data = None):
        """
        Train_data needs to be passed again when using quantum kernel.
        """
        if self.classical:
            return self.model.predict(test_data)
        else:
            test_matrix = self.kernel.evaluate(x_vec = test_data, y_vec = train_data) 
            return self.model.predict(test_matrix)
