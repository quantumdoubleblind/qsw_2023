import json
from datetime import datetime
import numpy as np
import os


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Python2JSON:
    def __init__(self, filename, hardware, experiment_nr, problem_size,
                 gate_keys, graph_densities, qaoa_layers, cmap_density, n_rows, n_cols, opt_level, comp_averages,
                 input_problem, times_ext, times_ext_stds, depth_ext, depth_ext_stds, gate_counts_ext,
                 gate_counts_ext_stds, times, depths, gate_counts, swap_gates):
        self.filename = filename
        self.hardware = hardware
        self.experiment_nr = experiment_nr
        self.problem_size = problem_size
        self.gate_keys = gate_keys
        self.graph_densities = graph_densities
        self.qaoa_layers = qaoa_layers
        self.cmap_densities = cmap_density
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.opt_level = opt_level
        self.comp_averages = comp_averages
        self.input_problem = input_problem
        self.times_ext = times_ext
        self.times_ext_stds = times_ext_stds
        self.depth_ext = depth_ext
        self.depth_ext_stds = depth_ext_stds
        self.gate_counts_ext = gate_counts_ext
        self.gate_counts_ext_stds = gate_counts_ext_stds
        self.times = times
        self.depths = depths
        self.gate_counts = gate_counts
        self.swap_gates = swap_gates

    def write2json(self):

        experiment_start = {}
        experiment = {}
        input_param = {}
        result = {}
        times = json.dumps(self.times)
        depths = json.dumps(self.depths)
        gate_counts = json.dumps(self.gate_counts)
        swap_gates = json.dumps(self.swap_gates)
        times_ext = json.dumps(self.times_ext)
        times_ext_stds = json.dumps(self.times_ext_stds)
        depth_ext = json.dumps(self.depth_ext)
        depth_ext_stds = json.dumps(self.depth_ext_stds)
        gate_counts_ext = json.dumps(self.gate_counts_ext)
        gate_counts_ext_stds = json.dumps(self.gate_counts_ext_stds)
        graph_densities = json.dumps(self.graph_densities.tolist())
        qaoa_layers = json.dumps(self.qaoa_layers.tolist())
        gate_keys = json.dumps(self.gate_keys)
        cmap_densities = json.dumps(self.cmap_densities.tolist())
        experiment['hardware_type'] = self.hardware
        experiment['input_problem'] = self.input_problem
        input_param['problem_size'] = self.problem_size
        input_param['gate_keys'] = gate_keys
        input_param['graph_density'] = graph_densities
        input_param['qaoa_layer'] = qaoa_layers
        input_param['cmap_densities'] = cmap_densities
        input_param['n_rows'] = self.n_rows
        input_param['n_cols'] = self.n_cols
        input_param['opt_level'] = self.opt_level
        input_param['comp_averages'] = self.comp_averages
        result['times_ext'] = times_ext
        result['times_ext_stds'] = times_ext_stds
        result['depth_ext'] = depth_ext
        result['depth_ext_stds'] = depth_ext_stds
        result['gate_counts_ext'] = gate_counts_ext
        result['gate_counts_ext_stds'] = gate_counts_ext_stds
        result['times'] = times
        result['depths'] = depths
        result['gate_counts'] = gate_counts
        result['swap_gates'] = swap_gates

        experiment_start['experiment_' + str(datetime.now())] = experiment
        experiment_start['input_parameter'] = input_param
        experiment_start['results'] = result

        json_object = json.dumps(experiment_start, cls=NumpyEncoder, indent=4)

        if self.startup_check():
            with open(str(self.filename) + str(self.experiment_nr), "w") as outputfile:
                outputfile.write(json_object)
        else:
            with open(str(self.filename) + str(self.experiment_nr), "w") as outputfile:
                outputfile.write(json_object)

    def startup_check(self):
        if os.path.exists(str(self.filename) + str(self.experiment_nr)):
            return True
        else:
            return False


def read_json(filename):
    f = open(filename)
    return json.load(f)
