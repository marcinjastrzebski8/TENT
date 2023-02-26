from qiskit import (ClassicalRegister, QuantumRegister, QuantumCircuit)
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt

"""
I am making estimation_circuit into a parametrised circuit so that I can use it in qiskit's QuantumKernel.
I believe I will only have to keep the forward U and the Kernel method goes backwards. - Yes, not needed.
In QuantumKernel.construct_circuit(), the inverse is being appended automatically using 
QuantumCircuit.append(inverse.to_instruction(),qubits) where the to_instruction() tells on which qubits to put the circuit
"""

def feature_map(nqubits, U, show=False):
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

    if show:
        circuit.decompose().draw('mpl')
        plt.show()


    return circuit
