import unittest
import numpy as np

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def projected_xyz_embedding(feature_map_circ, x):
    """
    Calculates the xyz expectation values for one data point,
    needed to find the frobenious norm as per L6 in 'Power of data...'. 

    feature_map: takes classical point to a quantum state
    x: point for which the projection is to be computed
    """
    estimator = Estimator()

    feature_mapped_x = feature_map_circ.assign_parameters(x)

    exp_vals = []
    for qubit_id in range(feature_map_circ.num_qubits):
        for pauli in ['X','Y','Z']:
            observable_as_list = ['I' for i in range(feature_map_circ.num_qubits)]
            #find expectation value for a given qubit for a given pauli
            observable_as_list[-(qubit_id+1)] = pauli
            observable_str = ''.join(observable_as_list)

            obs = SparsePauliOp(observable_str)
            job1 = estimator.run(feature_mapped_x, obs)

            exp_x = job1.result().values[0]
            exp_vals.append(exp_x)
    return exp_vals

def proj_xyz_data(feature_map_circ, X: np.ndarray) -> np.ndarray:
    """
    Calculates the xyz_embedding for a data set
    This can then be used in L6 in 'Power of data...'

    Returns an array whose shape is (np.shape(X)[0], np.shape(X)[1]*3*np.shape(X)[0]) (*3*np.shape(X)[0] from X,Y,Z expectations for each qubit assuming as many qubits as features)
    """
    X_proj = np.array([projected_xyz_embedding(feature_map_circ, x) for x in X])

    return X_proj

class TestProjection(unittest.TestCase):
    """
    Tests to ensure the projections give results as expected
    """
    def test_embedding(self):
        #this circuit will be used to create a Bell state and a |+>|0> state
        simple_circuit = QuantumCircuit(2)
        param = Parameter('theta')
        simple_circuit.h(0)
        simple_circuit.crx(param,0,1)
        x1 = [0]
        x2 = [np.pi]
        exps_x1 = projected_xyz_embedding(simple_circuit, x1)
        exps_x2 = projected_xyz_embedding(simple_circuit, x2)
        for (elem1, elem_test) in zip(exps_x1, [1,0,0,0,0,1]): self.assertAlmostEqual(elem1, elem_test)
        for elem2 in exps_x2: self.assertAlmostEqual(elem2, 0)

    

    
if __name__ == '__main__':
    unittest.main()
    