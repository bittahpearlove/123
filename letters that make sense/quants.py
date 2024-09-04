from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
from qiskit import transpile
from qiskit_aer import Aer
qc = QuantumCircuit(3)
qc.cx(0, 1)
qc.h(0),qc.h(1),qc.h(2) #i like to do this because it lead to random qubit results
qc.measure_all()
print(qc.draw())
simulator = Aer.get_backend('statevector_simulator')
statevector = simulator.run(transpile(qc)).result().get_statevector()
plot_bloch_multivector(statevector)