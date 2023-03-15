# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

import multiprocessing
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import FakeBrooklyn
import tsp
import qaoa
import number_partitioning
import maxcut
import max3sat
from qiskit import transpile
import time
import os
import sys
import python2json

nb_dir = os.getcwd()
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
from maxcut import *
from TopologyFunctions import *
from multiprocessing import Pool


class Experiment:

    def __init__(self, graph_density, opt_level, comp_averages, threads, problem, qaoa_layer):
        self.filename = "experiment"
        self.experiment_nr = "1"
        self.hardware = "use of fake backend brooklyn"
        self.problem = problem
        self.problem_type = problem
        self.initial = [0.1, 0.1]
        self.backend = FakeBrooklyn()
        self.config = self.backend.configuration()
        self.gate_set = self.config.basis_gates
        self.gate_keys = ['rz', 'sx', 'x', 'cx']
        self.gate_keys_swap = ['rz', 'sx', 'x', 'cx', 'swap']
        self.problem_sizes = np.array([3, 4, 5, 6, 7, 8, 9, 10])
        self.graph_densities = np.array([float(graph_density)])
        self.qaoa_layers = np.array([int(qaoa_layer)])
        self.cmap_densities = np.array(
           [0.013895, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
             0.9, 1.0])
        self.n_rows = 6
        self.n_cols = 3
        self.cmap = create_heavy_hex_IBMQ(self.n_rows, self.n_cols)
        self.opt_level = int(opt_level)
        self.comp_averages = int(comp_averages)
        self.parallel_threads = int(threads)

    def find_active_qubits(self, circuit):

        dag = circuit_to_dag(circuit)
        active_qubits = [qubit.index for qubit in circuit.qubits
                         if qubit not in dag.idle_wires()]

        return active_qubits

    def get_qubo_parallel(self, problem_sizes):

        depths = []
        depths_ext = np.zeros((len(self.problem_sizes), len(self.graph_densities), len(self.qaoa_layers),
                               len(self.cmap_densities)))
        depth_ext_stds = np.zeros((len(self.problem_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                   len(self.cmap_densities)))
        times = []

        times_ext = np.zeros((len(self.problem_sizes), len(self.graph_densities), len(self.qaoa_layers),
                              len(self.cmap_densities)))
        time_ext_stds = np.zeros((len(self.problem_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                  len(self.cmap_densities)))
        gate_counts = []
        gate_counts_ext = np.zeros((len(self.problem_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                    len(self.cmap_densities), 4))
        gate_count_ext_stds = np.zeros((len(self.problem_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                        len(self.cmap_densities), 4))
        swap_count = []

        if self.problem_type == "tsp":
            self.problem_sizes = [3]
            tsp_1 = tsp.TSP(2, 1)
            density_matrix, G = tsp_1.generate_problem_symm(problem_sizes, 0.7)
            qubo = tsp_1.generate_qubo(density_matrix, G)
        elif self.problem_type == "num_part":
            self.problem_sizes = np.array([3, 4, 6, 9, 16, 25, 36, 49, 64, 81, 100])
            num_p = number_partitioning.NumberPartitioning(problem_sizes)
            qubo = num_p.generate_qubo(num_p.generate_problems(1, problem_sizes)[0])  # eins zu loop
        elif self.problem_type == "max3sat":
            self.problem_sizes = np.array([6, 9, 16, 25, 36, 49, 64, 81, 100])
            max = max3sat.Max3Sat(3)
            qubo = max.generate_qubo(max.generate_problems(1, (3, problem_sizes))[0])  # eins zu loop
        elif self.problem_type == "maxcut":
            self.problem_sizes = np.array([3, 4, 6, 9, 16, 25, 36, 49, 64, 81, 100])
            G = maxcut.generate_graph_from_density(problem_sizes, 0.7)
            qubo = get_qubo_maxcut(G)
        else:
            return "problem type not defined yet"

        H = qubo

        theta = np.repeat(self.initial, 1)
        qc_qaoa = qaoa.get_qaoa_circ_from_qubo(H, theta, offset=0)

        for cd in range(len(self.cmap_densities)):
            cmap_density = self.cmap_densities[cd]
            print('coupling density: ', cmap_density)

            cmap_ext = increase_coupling_density(self.cmap, cmap_density)

            depths_qiskit_tmp = []
            times_tmp = []
            gate_count_tmp = []
            for _ in range(self.comp_averages):
                swap_count_tmp = []
                start = time.time()
                transpiled_circ = transpile(qc_qaoa, basis_gates=self.gate_set, coupling_map=cmap_ext,
                                            optimization_level=self.opt_level, approximation_degree=0.0)
                transpiled_circ_2 = transpile(qc_qaoa, basis_gates=self.gate_keys_swap,
                                              coupling_map=cmap_ext,
                                              optimization_level=self.opt_level)
                end = time.time()
                times_tmp.append(end - start)
                depths_qiskit_tmp.append(transpiled_circ.depth())
                gate_count_tmp.append(list(
                    transpiled_circ.count_ops()[key] if key in transpiled_circ.count_ops().keys() else 0 for
                    key in self.gate_keys))
                try:
                    swap_count_tmp.append(transpiled_circ_2.count_ops()['swap'])
                except Exception as e:
                    swap_count_tmp.append(0)

                times.append(times_tmp)
                times_ext[np.where(self.problem_sizes == problem_sizes)[0][0], 0, 0, cd] = np.mean(times_tmp)
                time_ext_stds[np.where(self.problem_sizes == problem_sizes)[0][0], 0, 0, cd] = np.std(times_tmp)
                depths.append(depths_qiskit_tmp)
                depths_ext[np.where(self.problem_sizes == problem_sizes)[0][0], 0, 0, cd] = np.mean(depths_qiskit_tmp)
                depth_ext_stds[np.where(self.problem_sizes == problem_sizes)[0][0], 0, 0, cd] = np.std(
                    depths_qiskit_tmp)
                gate_counts.append(gate_count_tmp)
                gate_counts_ext[np.where(self.problem_sizes == problem_sizes)[0][0], 0, 0, cd, :] = np.mean(
                    gate_count_tmp, axis=0)
                gate_count_ext_stds[np.where(self.problem_sizes == problem_sizes)[0][0], 0, 0, cd, :] = np.std(
                    gate_count_tmp, axis=0)
                swap_count.append(swap_count_tmp)

                lock = multiprocessing.Lock()
                lock.acquire()

                p2j = python2json.Python2JSON(
                    self.filename + "_" + str(0) + "_" + str(0) + "_" + str(problem_sizes) + "_" + str(cd) + "_",
                    self.hardware, self.experiment_nr, problem_sizes,
                    self.gate_keys, self.graph_densities[0], self.qaoa_layers,
                    cmap_density, self.n_rows, self.n_cols, self.opt_level,
                    self.comp_averages, self.problem, times_ext.tolist(),
                    time_ext_stds.tolist(), depths_ext.tolist(),
                    depth_ext_stds.tolist(), gate_counts_ext.tolist(),
                    gate_count_ext_stds.tolist(), times, depths, gate_counts, swap_count)
                p2j.write2json()

    def experiments(self):

        for i in range(len(self.qaoa_layers)):
            p = self.qaoa_layers[i]
            print('qaoa layers: ', p)
            for d in range(len(self.graph_densities)):
                density = self.graph_densities[d]
                print('graph density: ', density)
                with Pool(self.parallel_threads) as t:
                    t.map(self.get_qubo_parallel, self.problem_sizes)


if __name__ == "__main__":
    expr = Experiment(sys.argv[1],  sys.argv[2],  sys.argv[3],  sys.argv[4],  sys.argv[5],  sys.argv[6])

    e_name = "tsp_parallel"
    expr.experiments()
    try:
        print("im here")
        expr.experiments()
    except Exception as e:
        f = open("errorlog.txt", "a")
        f.write(str(e))
        f.close()
