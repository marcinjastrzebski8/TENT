from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
#STATUS: FIRST ONE WORKS FINE BUT IS NOW OBSOLETE. SECOND WORKS FINE AND CURRENTLY USED.
#THOUGHT: U_PARAM SHOULD BE MADE INTO AN ABSTRACT CLASS WHERE TYPES OF MAPS ARE SHARED (SOME GENERAL LIBRARY)
#BUT THE CIRCUIT ITSELF WOULD DEFINE A CLASS WHICH CAN BE INSTANCIATED. 
#ADDITIONALLY, THOSE DERIVED CLASSES SHOULD BE UNIT TESTED [BEFORE PUBLICATION???]


def U_flexible(nqubits,params,single_mapping=0,pair_mapping=0,interaction = 'ZZ', alpha = 1, draw = False):
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
    if draw:
        circuit.draw('mpl')
        plt.show()
    return circuit

"""
#For Simeon to see:
from qiskit.circuit import ParameterVector
params = ParameterVector('phi',4) 
circ_to_test = U_flexible(4 ,params,single_mapping = 1 ,pair_mapping = 1,interaction = 'YY', alpha = 1, draw = True)
"""