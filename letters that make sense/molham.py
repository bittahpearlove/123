import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
import seaborn as sns

symbols = ['H', 'H']
coordinates = np.array([[-0.673, 0, 0], [0.673, 0, 0]])
H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)
print("Number of qubits:", qubits)
print("Hamiltonian:\n", H)

dev = qml.device('default.qubit', wires=qubits)
@qml.qnode(dev)
def expen(state):
    obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    qml.Hamiltonian(coordinates, obs)
    qml.BasisState(np.array(state), wires=range(qubits))
    return qml.expval(H)

hf = qchem.hf_state(electrons=2, orbitals=qubits)
expectation_value_1 = expen([1, 0, 1, 0])
expectation_value_hf = expen(hf)

def plot_molecule(coordinates, symbols):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=100, color='b')

    for i, (x, y, z) in enumerate(coordinates):
        ax.text(x, y, z, symbols[i], size=20, zorder=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Molecular Structure of H₂')
    plt.show()
plot_molecule(coordinates, symbols)
print("Expectation value for state [1, 0, 1, 0]:", expectation_value_1)
print("Expectation value for Hartree-Fock state:", expectation_value_hf)