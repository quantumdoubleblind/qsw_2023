import multiprocessing

import qiskit.transpiler
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

    def __init__(self, problem):
        self.filename = "max_cut_parallel_backend"
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
        self.problem_sizes = np.array([100])
        self.graph_densities = np.array([0.7])
        self.qaoa_layers = np.array([1])
        self.cmap_densities = np.array(
            [0.013895, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
             0.9, 1.0])
        self.n_rows = np.array([6, 6, 8, 6, 8, 8])
        self.n_cols = np.array([4, 5, 4, 6, 5, 6])
        self.backend_sizes = [[6, 4], [6, 5], [8, 4], [6, 6], [8, 5], [8, 6]]
        self.opt_level = 1
        self.comp_averages = 20

    def find_active_qubits(self, circuit):

        dag = circuit_to_dag(circuit)
        active_qubits = [qubit.index for qubit in circuit.qubits
                         if qubit not in dag.idle_wires()]

        return active_qubits

    def get_qubo_parallel(self, backend_sizes):

        depths = []
        depths_ext = np.zeros((len(self.backend_sizes), len(self.graph_densities), len(self.qaoa_layers),
                               len(self.cmap_densities)))
        depth_ext_stds = np.zeros((len(self.backend_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                   len(self.cmap_densities)))
        times = []

        times_ext = np.zeros((len(self.backend_sizes), len(self.graph_densities), len(self.qaoa_layers),
                              len(self.cmap_densities)))
        time_ext_stds = np.zeros((len(self.backend_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                  len(self.cmap_densities)))
        gate_counts = []
        gate_counts_ext = np.zeros((len(self.backend_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                    len(self.cmap_densities), 4))
        gate_count_ext_stds = np.zeros((len(self.backend_sizes), len(self.graph_densities), len(self.qaoa_layers),
                                        len(self.cmap_densities), 4))
        swap_count = []

        if self.problem_type == "tsp":
            tsp_1 = tsp.TSP(2, 1)
            density_matrix, G = tsp_1.generate_problem_symm(self.problem_sizes[0], 0.7)
            qubo = tsp_1.generate_qubo(density_matrix, G)
        elif self.problem_type == "num_part":
            num_p = number_partitioning.NumberPartitioning(self.problem_sizes[0])
            qubo = num_p.generate_qubo(num_p.generate_problems(1, self.problem_sizes[0])[0])  # eins zu loop
        elif self.problem_type == "max3sat":
            max = max3sat.Max3Sat(3)
            qubo = max.generate_qubo(max.generate_problems(1, (3, self.problem_sizes[0]))[0])  # eins zu loop
        elif self.problem_type == "maxcut":
            G = maxcut.generate_graph_from_density(self.problem_sizes[0], 0.7)
            qubo = get_qubo_maxcut(G)
        else:
            return "problem type not defined yet"

        cmap = create_heavy_hex_IBMQ(backend_sizes[0], backend_sizes[1])
        H = qubo

        theta = np.repeat(self.initial, 1)
        qc_qaoa = qaoa.get_qaoa_circ_from_qubo(H, theta, offset=0)

        for cd in range(len(self.cmap_densities)):
            cmap_density = self.cmap_densities[cd]
            print('coupling density: ', cmap_density)

            cmap_ext = increase_coupling_density(cmap, cmap_density)

            depths_qiskit_tmp = []
            times_tmp = []
            gate_count_tmp = []
            count = 0
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
                times_ext[self.backend_sizes.index(backend_sizes), 0, 0, cd] = np.mean(times_tmp)
                time_ext_stds[self.backend_sizes.index(backend_sizes), 0, 0, cd] = np.std(times_tmp)
                depths.append(depths_qiskit_tmp)
                depths_ext[self.backend_sizes.index(backend_sizes), 0, 0, cd] = np.mean(depths_qiskit_tmp)
                depth_ext_stds[self.backend_sizes.index(backend_sizes), 0, 0, cd] = np.std(depths_qiskit_tmp)
                gate_counts.append(gate_count_tmp)
                gate_counts_ext[self.backend_sizes.index(backend_sizes), 0, 0, cd, :] = np.mean(gate_count_tmp, axis=0)
                gate_count_ext_stds[self.backend_sizes.index(backend_sizes), 0, 0, cd, :] = np.std(gate_count_tmp,
                                                                                                   axis=0)
                swap_count.append(swap_count_tmp)

                lock = multiprocessing.Lock()
                lock.acquire()

                p2j = python2json.Python2JSON(
                    self.filename + "_" + str(0) + "_" + str(0) + "_" + str(backend_sizes) + "_" + str(cd) + "_",
                    backend_sizes, self.experiment_nr, self.problem_sizes,
                    self.gate_keys, self.graph_densities[0], self.qaoa_layers,
                    cmap_density, self.n_rows, self.n_cols, self.opt_level,
                    self.comp_averages, self.problem, times_ext.tolist(),
                    time_ext_stds.tolist(), depths_ext.tolist(),
                    depth_ext_stds.tolist(), gate_counts_ext.tolist(),
                    gate_count_ext_stds.tolist(), times, depths, gate_counts, swap_count)
                p2j.write2json()
                count += 1

    def experiments(self):

        for i in range(len(self.qaoa_layers)):
            p = self.qaoa_layers[i]
            print('qaoa layers: ', p)
            for d in range(len(self.graph_densities)):
                density = self.graph_densities[d]
                print('graph density: ', density)
                with Pool(128) as t:
                    t.map(self.get_qubo_parallel, self.backend_sizes)


if __name__ == "__main__":
    problem_type = "maxcut"
    expr = Experiment(problem_type)

    e_name = "maxcut_parallel_larger_backend"
    expr.experiments()
    try:
        expr.experiments()
    except Exception as e:
        f = open("errorlog.txt", "a")
        f.write(str(e))
        f.close()
