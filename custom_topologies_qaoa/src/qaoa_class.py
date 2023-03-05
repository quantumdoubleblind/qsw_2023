# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

import numpy as np
import itertools
from qiskit import QuantumCircuit, Aer
import math
from scipy.optimize import minimize
import time
from qiskit.circuit import Parameter
from qiskit_optimization import QuadraticProgram
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class standard_QAOA:
    # defines an n-qubit instance of standard QAOA circuit for a QUBO matrix Q ith p layers. Optional argument: Precompiled circuit
    def __init__(self, Q, p, circ=None):
        n = len(Q)
        if circ:
            self.qc = circ
        if not circ:
            self.qc = QuantumCircuit(n)
        self.parameters = []
        self.n = n
        self.p = p

    # define circuit with p = number of layers
    def qaoa_circuit(self, Q):

        # order of parameters: [gamma_0, beta_0, gamma_1, beta_1, ...]

        # index tuples for 2-qubit gates: All possible combinations [i,j] for i!=j and i,j \in {1,num_qubits} whereas [i,j] = [j,i]
        tq_id = list(itertools.combinations(np.linspace(0, self.n - 1, self.n, dtype=int), 2))

        """
        Convert Hamiltonian directly to Ising model to calculate the coefficients of the Pauli-Terms. 
        Contains only linar and quadratic terms (2D qubo-matrix), resulting in single- and two-qubit rotation gates. 
        Ising Hamiltonian: H = \Sum_i h_i * s_i + \Sum_{i<j} J_{ij} s_i s_j + C
        Qubo Hamiltonian: H = x^T Q x + offset
        """
        Q = Q / np.max(np.abs(Q))  # normalize QUBO

        # Linear ising terms: Combinations with one Pauli-Z and I otherwise. 
        Ising_h = []
        for i in range(self.n):
            Ising_h.append(0.5 * Q[i, i] + 0.25 * np.sum([(Q[i, j] + Q[j, i]) for j in range(self.n) if j != i]))

        # quadratic Ising terms: Use J[i,j] only for unequal indices [[i,j] 
        Ising_J = 0.25 * Q

        # dictionaries: {index, the indices of the qubit(s) to which Z-operators are applied (all other get identities), Ising-coefficient}
        sq_gates = {}
        tq_gates = {}
        for j in range(self.n):
            sq_gates[j] = j, Ising_h[j]
        for j in range(len(tq_id)):
            tq_gates[j] = tq_id[j], Ising_J[tq_id[j][0], tq_id[j][1]]

        # Intial state: Eigenstate of mixing Hamiltonian
        for i in range(0, self.n):
            self.qc.h(i)

        # QAOA layers
        for irep in range(0, self.p):

            # Problem Hamiltonian 
            self.parameters.append(Parameter("γ" + "_" + str(irep)))
            for key in tq_gates.keys():  # two-qubit gates
                self.qc.rzz(self.parameters[2 * irep] * tq_gates[key][1], tq_gates[key][0][0], tq_gates[key][0][1])

            for key in sq_gates.keys():  # single-qubit gates
                self.qc.rz(self.parameters[2 * irep] * sq_gates[key][1], sq_gates[key][0])

            # Mixing Hamiltonian
            self.parameters.append(Parameter("β" + "_" + str(irep)))
            for i in range(0, self.n):
                self.qc.rx(self.parameters[2 * irep + 1], i)

        # important for execution. Adds the apropriate number of classical bits automatically to store the measurement outcome
        self.qc.measure_all()

    def compute_energy(self, params, Q, offset, backend, nshots, seed):
        # bind the parameters to the circuit
        values = dict(zip(self.parameters, params))
        circ = self.qc.bind_parameters(values)

        # run the circuit with the parameters passed in params
        job = backend.run(circ, seed_simulator=seed, shots=nshots)
        counts = job.result().get_counts()

        mean_energy = 0
        for k in counts:
            bitstring = np.array([int(x) for x in k])
            # evaluate bitstring k for the QUBO matrix Q with numpy matrix multiplication
            mean_energy += (bitstring @ Q @ bitstring + offset) * counts[k]
        mean_energy /= sum([counts[k] for k in counts])

        return mean_energy

        # tune parameters num_iterations times

    # return parameters after num_iterations iterations
    def optimize_params(self, Q, offset, initial_params, optimizer, backend, nshots, seed, num_iterations):
        func = lambda x: self.compute_energy(x, Q, offset, backend, nshots, seed)
        theta = minimize(func,
                         x0=initial_params,
                         method=optimizer,
                         options={"maxiter": num_iterations})

        return theta


def execute_qaoa(Q, offset, p, initial_params, optimizer, backend, nshots):
    seed = 123
    results = dict()
    qaoa = standard_QAOA(Q, p)
    qaoa.qaoa_circuit(Q)

    start = time.time()
    res = qaoa.optimize_params(Q, offset, initial_params, optimizer, backend, nshots, seed, num_iterations=1000)
    opt_time = time.time() - start

    opt_params = res.x

    # create circuit with final parameters
    values = dict(zip(qaoa.parameters, opt_params))
    circ = qaoa.qc.bind_parameters(values)

    # run the circuit with the final parameters
    job = backend.run(circ, seed_simulator=seed, shots=nshots)
    counts = job.result().get_counts()

    expvals = {}
    bitstrings = {}
    it = 0
    for k in counts:
        bitstring = np.array([int(x) for x in k])
        expvals[it] = (bitstring @ Q @ bitstring + offset) * counts[k]
        bitstrings[it] = bitstring
        it += 1

    id_min = sorted(expvals.items(), key=lambda item: item[1])[0][0]

    results["sol"] = bitstrings[id_min]
    results["fval"] = res.fun
    results["mean_energy"] = expvals[id_min]
    results["counts"] = counts
    results["optimal_params"] = opt_params.tolist()
    results["optimizer_iter"] = res.nfev
    results["optimizer_time"] = opt_time

    return results


def solve_qaoa_qiskit(H, offset, nshots, seed, backend, optimizer, p, initial_params=None):
    algorithm_globals.random_seed = seed
    qins = QuantumInstance(backend=Aer.get_backend(backend), shots=nshots, seed_simulator=seed, seed_transpiler=seed)

    qp = QuadraticProgram()
    for i in range(len(H)):
        qp.binary_var('x{0}'.format(i))
    qp.minimize(constant=offset, linear=np.diag(H), quadratic=np.triu(H, 1))

    if len(initial_params) == 0:
        initial_params = (-np.pi + 2 * np.pi * np.random.rand(p, 2)).flatten()

    # Initialize QAOA
    qins = QuantumInstance(backend=Aer.get_backend(backend), shots=nshots, seed_simulator=seed, seed_transpiler=seed)
    qaoa_mes = QAOA(optimizer=optimizer, reps=p, include_custom=True, quantum_instance=qins,
                    initial_point=initial_params)
    qaoa = MinimumEigenOptimizer(qaoa_mes)

    # Solve QAOA
    start = time.time()
    qaoa_result = qaoa.solve(qp)
    end = time.time()
    elapsed_time = end - start
    print(qaoa_result)

    return (qaoa_result, elapsed_time, initial_params)
