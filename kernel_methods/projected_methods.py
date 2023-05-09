from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def projected_xyz_embedding(feature_map_circ, x):
    """
    Calculates the xyz expectation values for one data point,
    needed to find the frobenious norm as per L6 in 'Power of data...'. 

    feature_map: takes classical point to a quantum state
    x: point for which the projection is to be computed
    """
    estimator = Estimator()
    observable_as_list = ['I' for i in range(len(x))]

    feature_mapped_x = feature_map_circ.assign_parameters(x)

    exp_vals = []
    for qubit_id in range(len(x)):
        for pauli in ['X','Y','Z']:
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

    Returns an array whose shape is (np.shape(X)[0], np.shape(X)[1]*3) (*3 from X,Y,Z expectations)
    """
    X_proj = np.array([projected_xyz_embedding(feature_map_circ, x) for x in X])

    return X_proj
