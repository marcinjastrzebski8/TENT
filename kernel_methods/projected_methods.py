from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def frob_norm_1rdm(feature_map_circ, x1, x2, qubit_id):
    """
    Calculates the frobenious norm as per L6 in 'Power of data...'
    for a reduced density matrix of a qubit.
    Utilises exp values of Paulis. 

    feature_map: takes classical point to a quantum state
    x1, x2: two points for which the kernel is to be computed
    qubit: index of the qubit for which the rdm is calculated
    """
    estimator = Estimator()
    observable_as_list = ['I' for i in range(len(x1))]

    feature_mapped_x1 = feature_map_circ.assign_parameters(x1)
    feature_mapped_x2 = feature_map_circ.assign_parameters(x2)

    frob_norm = 0
    for pauli in ['X','Y','Z']:
        observable_as_list[-(qubit_id+1)] = pauli
        observable_str = ''.join(observable_as_list)

        obs1 = SparsePauliOp(observable_str)
        job1 = estimator.run(feature_mapped_x1, obs1)
        job2 = estimator.run(feature_mapped_x2, obs1)

        exp_x1 = job1.result().values[0]
        exp_x2 = job2.result().values[0]

        rdm_distance = (exp_x1-exp_x2)**2
        frob_norm += rdm_distance
    return frob_norm

def proj_kernel_1rdm(feature_map_circ, x1, x2, gamma):
    """
    Calculates the projected one-particle reduced density matrix kernel.
    x1, x2: data points
    L6 in 'Power of data...'
    Note that BaseStateFidelity class exists in qiskit.
    Maybe it'd be benefitial to implement this using it.
    """
    sum_over_qubits = 0
    for qubit_id in range(len(x1)):
        sum_over_qubits += frob_norm_1rdm(feature_map_circ, x1, x2, qubit_id)
    proj_kernel_val = np.exp(-gamma*sum_over_qubits)

    return proj_kernel_val
