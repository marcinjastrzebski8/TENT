"""
Use data provided to train and save an svm classifier
Can use a classicla svm or quantum-enhanced
STATUS: in dev, job report could be compiled in main
"""

import numpy as np
import sys
import os
from typing import List, Optional
import joblib
from functools import reduce
import time
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel


from qiskit import IBMQ
IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(group='open')
from qiskit.circuit.library import PauliFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel, QuantumKernel, BaseKernel
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import QSVC
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

from . import projected_methods

def self_product(x: np.ndarray) -> float:
    """
    This function ensures the phi(x) in the circuit matches the one I've been using
    when I first coded this myself.
    Only needed for equal comparison to already existing results.

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = -x[0] if len(x) == 1 else -1*reduce(lambda m, n: m * n, (np.pi-x))
    return coeff


class ProjectedKernel1RDM(BaseKernel):
    """
    proj_kernel_1rdm uses the to-be-deprecated QuantumKernel.
    In the future will need to change that to use Sampler/Estimator primitives.
    """
    def __init__(self, feature_map, gamma):
        super().__init__(feature_map=feature_map)
        self.gamma = gamma
    def evaluate(self, x1: np.ndarray, x2: np.ndarray = None) -> np.ndarray:
        x1, x2 = self._validate_input(x1, x2)
        #determine if calculating self inner product
        is_symmetric = True
        if x2 is None:
            x2 = x1
        elif not np.array_equal(x1, x2):
            is_symmetric = False

        kernel_shape = (x1.shape[0], x2.shape[0])
        kernel_matrix = np.zeros(kernel_shape)
        #TODO!!!!!!: CHECK if I can parallelise this (shared-memory, within each node)
        #BUT NOTE PROFILER WILL NOT WORK PROLLY (WOULD NEED PARALLEL PROFS)
        #SO SHOULD DO PROFILING ON ONE THREAD, THEN PARALLELISE
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                #unfortunately each estimation atm takes almost 1s, makes for very long kernel-finding
                if i == j: kernel_matrix[i][j] = 1.
                elif (is_symmetric) and (j>i):
                    kernel_matrix[i][j] = projected_methods.proj_kernel_1rdm(self._feature_map, x1[i], x2[j], self.gamma)
                    kernel_matrix[j][i] = kernel_matrix[i][j]
                elif (is_symmetric) and (i>j): pass
                else:
                    kernel_matrix[i][j] = projected_methods.proj_kernel_1rdm(self._feature_map, x1[i], x2[j], self.gamma)
        return kernel_matrix



class QKE_SVC():
    """
    Defines an SVC classifier model family - either classical or quantum-enhanced.
    Can be used to train/load a given instance of the family.
    Classical case is an rbf.
    Quantum is Havlicek-based (IPQ).
    """
    def __init__(self,
    kernel_type: str,
    class_weight, 
    gamma_class = None, 
    gamma_quant = None,
    C_class = None, 
    alpha = None,
    C_quant = None,
    paulis: Optional[List] = None,
    circuit_width = None,
    keep_kernel = False):
        if kernel_type in ('classical_keep_kernel', 'classical'):
            self.gamma = gamma_class
            self.C_class = C_class
        else:
            #TODO:reshuffle these into options for original and projected methods
            self.alpha = alpha
            self.C_quant = C_quant
            self.backend = QuantumInstance(AerSimulator(method = 'statevector'))
            self.circuit_width = circuit_width
            self.paulis = paulis
            self.featureMap = PauliFeatureMap(circuit_width, alpha = self.alpha, paulis = self.paulis, data_map_func = self_product)
            if kernel_type == 'fidelity':
                self.kernel = QuantumKernel(feature_map=self.featureMap, quantum_instance = self.backend)
            elif kernel_type == 'projected_1rdm':
                self.gamma = gamma_quant
                self.kernel = ProjectedKernel1RDM(feature_map = self.featureMap, gamma = self.gamma)

            """
            this implementation recommended from qiskit's migration guide
            using qiskit runtime I think makes it run in cloud so could be faster
            NOTE for now commenting out because new method is MUCH SLOWER
            service = QiskitRuntimeService(channel = 'ibm_quantum')
            backend = service.get_backend('simulator_statevector')
            sampler = Sampler(session=backend)
            fidelity = ComputeUncompute(sampler=sampler)
            self.kernel = FidelityQuantumKernel(feature_map=featureMap, fidelity = fidelity)
            """

        self.keep_kernel = keep_kernel
        self.kernel_type = kernel_type
        self.class_weight = class_weight
        self.cache_chosen = 1000

    def train_model(self, train_data, train_labels, from_config: str):
        """
        When using a precomputed kernel fit and predict differ.
        """
        if 'classical' in self.kernel_type:
            if not self.keep_kernel:
                #this is the original, clean implementation
                model = SVC(kernel = 'rbf', 
                gamma = self.gamma,
                C = self.C_class,
                cache_size = self.cache_chosen,
                class_weight = self.class_weight)

                model.fit(train_data, train_labels)
            else:
                #this way we can keep the kernel matrix
                #timing may differ from original
                model = SVC(kernel = 'precomputed',
                cache_size = self.cache_chosen,
                class_weight = self.class_weight)
                kernel_name = 'kernel_'+from_config

                train_matrix = rbf_kernel(train_data, gamma = self.gamma)
                kernel_path = str(Path().absolute() / 'saved_kernels' / kernel_name)
                np.save(kernel_path, train_matrix)
                model.fit(train_matrix, train_labels)
        else: #use quantum kernel estimation
            #changed from QSVC implementation
            model = SVC(kernel = 'precomputed',
                C = self.C_quant,
                cache_size = self.cache_chosen,
                class_weight = self.class_weight)
            if not self.keep_kernel:
                #clean method
                model.fit(train_data, train_labels)
            else:
                #likely that in the quantum case the two methods (keep_kernel True/False) take the same amount of time
                #but keeping both options just for completeness
                #from preliminary checks (one run on small data sample) this appears slower!
                train_matrix = self.kernel.evaluate(train_data)
                np.save('kernel_'+from_config, train_matrix)
                model.fit(train_matrix, train_labels)
        #save fitted SVC model
        filename = 'model_from_'+from_config+'.sav'
        model_path = str(Path().absolute() / 'sliced_detector_analysis' / 'saved_classifiers' / filename)
        joblib.dump(model, model_path)
        print('SVC model trained and stored as:', filename)
        return model

    def set_model(self, load, model = None, train_data = None, train_labels = None, from_config = None):
        if load:
            self.model = model
            print('model has been loaded, model: ', self.model)
        else:
            self.model = self.train_model(train_data = train_data, train_labels = train_labels, from_config = from_config)

    def test(self, test_data, train_data = None):
            if self.keep_kernel:
                if 'classical' in self.kernel_type:
                    test_matrix = rbf_kernel(test_data, train_data, gamma = self.gamma)
                elif self.kernel_type in ['fidelity', 'projected_1rdm']:
                    test_matrix = self.kernel.evaluate(test_data, train_data)
                else:
                    raise ValueError('Kernel type not supported')
                return self.model.predict(test_matrix)
            else:
                return self.model.predict(test_data)
