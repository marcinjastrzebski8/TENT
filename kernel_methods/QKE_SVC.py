"""
Use data provided to train and save an svm classifier
Can use a classicla svm or quantum-enhanced
STATUS: in dev, job report could be compiled in main
"""
import unittest
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
#for testing
from qiskit import QuantumCircuit, QuantumRegister 
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

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
                C = self.C_class,
                cache_size = self.cache_chosen,
                class_weight = self.class_weight)
                
                train_matrix = rbf_kernel(train_data, gamma = self.gamma)

                kernel_name = 'kernel_'+from_config
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

                kernel_name = 'kernel_'+from_config
                kernel_path = str(Path().absolute() / 'saved_kernels' / kernel_name)
                np.save(kernel_path, train_matrix)

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
                return self.model.predict(test_matrix), self.model.decision_function(test_matrix)
            else:
                return self.model.predict(test_data), self.model.decision_function(test_data)


class TestOldVsNew(unittest.TestCase):
    """
    Test to ensure old and new implementations of feature maps are the same
    """
    #copied from old version of the project, old method of implementing feature maps
    def U_flexible(self, nqubits,params,single_mapping=0,pair_mapping=0,interaction = 'ZZ', alpha = 1, draw = False):
            """
            U gate defines the feature map circuit produced by feature_map
            Applies a series of rotations parametrised by input data.
            From Havlicek et. al.
            circuit -> QuantumCircuit object to which U is attached - note: using .append() instead causes a qiskit bug to throw errors later
            params  -> ParameterVector objects, each parameter corredponds to a feature in a data point

            User can choose function for mapping
            """
            qbits = QuantumRegister(nqubits,'q')
            circuit = QuantumCircuit(qbits)

            #define some maps for single-qubit gates to choose from
            def single_map(param):
                if single_mapping == 0:
                    return param*nqubits
                elif single_mapping == 1:
                    return param
                elif single_mapping == 2:
                    return param*param
                elif single_mapping == 3:
                    return param*param*param #note ** does not work for qiskit ParameterVector element objects
                elif single_mapping == 4:
                    return param*param*param*param 


            #define some maps for two-qubit gates to choose from
            def pair_map(param1,param2):
                if pair_mapping == 0:
                    return param1*param2
                elif pair_mapping == 1:
                    return (np.pi-param1)*(np.pi-param2)
                elif pair_mapping == 2:
                    return (np.pi-(param1*param1))*(np.pi-(param2*param2))
                elif pair_mapping == 3:
                    return(np.pi-(param1*param1*param1))*(np.pi-(param2*param2*param2))
                elif pair_mapping == 4:
                    return(np.pi-(param1*param1*param1*param1))*(np.pi-(param2*param2*param2*param2))

            #use chosen single-qubit mapping to make a layer of single-qubit gates
            for component in range(nqubits):
                phi_j = single_map(params[component])
                circuit.rz(-2*alpha*phi_j,qbits[component])
            #use chosen two-qubit mapping to make a layer of 2-qubit gates
            for first_component in range(0,nqubits-1):
                for second_component in range(first_component+1,nqubits):
                    #Havlicek
                    #Note there was an mistake here when making results until 19/05/2022. last line was (qbits[0], qbits[component]) not sure how that even worked
                    #Note these are implemented to only use H, CX, X, Z (could just say 'rz1, rz2')
                    phi_ij = pair_map(params[first_component],params[second_component])
                    if interaction == 'ZZ':
                        circuit.cx(qbits[first_component],qbits[second_component])
                        circuit.rz(-2*alpha*phi_ij,qbits[second_component])
                        circuit.cx(qbits[first_component],qbits[second_component])
                        #Park
                    if interaction == 'YY': 
                        circuit.rx(np.pi/2,qbits[first_component])
                        circuit.rx(np.pi/2,qbits[second_component])
                        circuit.cx(qbits[first_component], qbits[second_component])
                        circuit.rz(-2*alpha*phi_ij, qbits[second_component])
                        circuit.cx(qbits[first_component], qbits[second_component])
                        circuit.rx(-np.pi/2, qbits[first_component])
                        circuit.rx(-np.pi/2, qbits[second_component])
                    if interaction == 'XX':
                        #get this from Simeon
                        pass
            return circuit
    
    def feature_map(self, nqubits, U, show=False):
        """
        Feature map circuit following Havlicek et al.
        nqubits  -> int, number of qubits, should match elements of input data
        U        -> gate returning QuantumCircuit object. Defines the feature map.
        """
        qbits = QuantumRegister(nqubits,'q')
        circuit = QuantumCircuit(qbits)
        #barriers just to make visualisation nicer
        circuit.h(qbits[:])
        circuit.barrier()
        #forward with x_i
        circuit.append(U.to_instruction(),circuit.qubits)
        circuit.barrier()
        circuit.h(qbits[:])
        circuit.barrier()
        circuit.append(U.to_instruction(),circuit.qubits)
        circuit.barrier()

        return circuit
    
    def test_feature_maps(self):
        param_vector = ParameterVector('phi', 9)
        random_features = np.random.rand(9)

        old_gate = self.U_flexible(9, param_vector, 1, 1, 'YY', alpha = 0.1)
        quant_old_map = self.feature_map(9, old_gate)

        quant_new = QKE_SVC('fidelity', {0:1, 1:1}, 1, 1, 1000000, 0.2, 1000000, ['Z', 'YY'], 9, True)
        quant_new_map = quant_new.featureMap

        quant_old_map = quant_old_map.assign_parameters(random_features)
        quant_new_map = quant_new_map.assign_parameters(random_features)

        old_sv = Statevector(quant_old_map)
        new_sv = Statevector(quant_new_map)

        self.assertEqual(old_sv.equiv(new_sv), True)

    def test_kernels(self):
        #to avoid copying large chunks of previous code I'm comparing to results obtained once, in an equivalent setup with the old code
        #NOTE this isn't ideal for other people using this repo; could copy results into here but they don't mean much to an external user?
        old_preds_quant = list(np.load('/Users/marcinjastrzebski/Desktop/ACADEMIA/FIRST_YEAR/TrackML/QuantumKernelEstimation/for_TENT_unittest_quant.npy'))
        old_preds_class = list(np.load('/Users/marcinjastrzebski/Desktop/ACADEMIA/FIRST_YEAR/TrackML/QuantumKernelEstimation/for_TENT_unittest_class.npy'))

        quant_new = QKE_SVC('fidelity', {0:1, 1:1}, 1, 1, 1000000, 0.2, 1000000, ['Z', 'YY'], 3, True)
        class_new = QKE_SVC('classical_keep_kernel', {0:1, 1:1}, 1, 1, 1000000, 0.2, 1000000, ['Z', 'YY'], 3, keep_kernel=True)
        np.random.seed(1008)
        X_train = np.random.rand(100,3)
        y_train = np.random.choice([0,1], 100)

        X_test = np.random.rand(50,3)
        y_test = np.random.choice([0,1], 50)

        quant_new.set_model(load = False, train_data = X_train, train_labels = y_train, from_config = 'unittest_quant')
        new_preds_quant = list(quant_new.test(X_test, X_train)[0])
        class_new.set_model(load = False, train_data = X_train, train_labels = y_train, from_config = 'unittest_class')
        new_preds_class = list(class_new.test(X_test, X_train)[0])



        self.assertListEqual(new_preds_quant, old_preds_quant)
        self.assertListEqual(old_preds_class, new_preds_class)
        
        
if __name__ == '__main__':
    unittest.main()
        