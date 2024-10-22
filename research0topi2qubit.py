from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere
import numpy as np
import pennylane as qml
import torch
import random
n = 2
dev = qml.device("default.qubit", wires=n)
@qml.qnode(dev, interface='torch')
def circuit(params):
    qml.RZ(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
def cost(params):
    return circuit(params)
params = torch.tensor([np.pi], requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.1)
for step in range(500):
    optimizer.zero_grad()
    loss = cost(params)
    loss.backward()
    optimizer.step() 
    if step % 10 == 0: 
        print(f"Step {step}, Loss: {loss.item()}, Params: {params.data.numpy()}")
print("Final parameters:", params.data.numpy())
qc = QuantumCircuit(n)
for i in range(n) :
  qc.h(i)
  qc.rz(float(params.data.numpy()),0)
sv = Statevector.from_instruction(qc).data
for index, amplitude in enumerate(sv):
  state = format (index, f'0{n}b')
  print(f'|{state}>: {amplitude: .4f}')
plot_state_qsphere(sv)
