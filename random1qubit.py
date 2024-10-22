import qiskit
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
from qiskit import transpile
from qiskit_aer import Aer
import random
import math
qc = qiskit.QuantumCircuit(1)
angle = random.uniform(0, 2 * math.pi)
inp = [math.cos(angle),math.sin(angle)]
qc.initialize(inp,0)
qc.h(0)
qc.z(0)
qc.measure_all()
simulator = Aer.get_backend('statevector_simulator')
statevector = simulator.run(transpile(qc)).result().get_statevector()
plot_bloch_multivector(statevector)
qc.draw()