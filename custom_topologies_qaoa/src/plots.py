# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

from python2json import read_json
import ast
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plots:

    def __init__(self, filename):
        self.filename = filename
        self.density = [0.013, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.density = ['0.013895', '0.02', '0.025', '0.03', '0.035', '0.04', '0.045', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6',
                        '0.7', '0.8', '0.9', '1.0']
        #self.density = [0.013895, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        #self.density = ['0.013895', '0.02', '0.025', '0.03', '0.035', '0.04', '0.045', '0.05']
        self.problem_sizes = [3, 4, 6, 9, 16, 25, 36, 49, 64, 81, 100]
        #self.problem_sizes = [9, 16, 25, 36, 49, 64, 81, 100]
        #self.problem_sizes = [6, 7, 8, 9, 16, 25, 36, 49, 64, 81, 100]
        #self.problem_sizes_string = ['9','16', '25', '36', '49', '64', '81', '100']
        self.problem_sizes_string = ['3', '4','6','9', '16', '25', '36', '49', '64', '81', '100']
        #self.problem_sizes_string = ['6','7', '8', '9', '16', '25', '36', '49', '64' , '81', '100']
        self.comp_averages = 20
        self.gate_keys = ['rz', 'sx', 'x', 'cx']
        self.colors = ['C9', 'm', 'C3', "C7", 'r', 'b', 'g', 'C0', 'C1', 'C8', 'C5', 'C6', 'C8']
        self.colors = ['cornflowerblue', 'mediumseagreen', 'indianred', 'bisque', 'orchid', 'turquoise', 'grey', 'gold',
                  'lawngreen', 'slateblue', 'darkorange']
        self.dark_colors = ['navy', 'darkgreen', 'maroon', 'darkorange', 'darkmagenta', 'darkcyan', 'black', 'peru',
                       'limegreen', 'indigo', 'grey']

        #self.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

    def flat_list(self, l):
        return [item for sublist in l for item in sublist]

    def plot_density_circuit_depth(self, plotname):
        data = read_json(self.filename)
        depth_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['depth_ext'])))

        x_axis = self.density
        #x_axis = [0.05, 0.1, 0.15]
        f, ax = plt.subplots()
        for i in range(0, len(self.problem_sizes_string)):
            y_axis = depth_ext[i]
            ax.plot(x_axis, y_axis)
        plt.title('circuit depth vs density')
        plt.xlabel('density')
        plt.ylabel('circuit_depth')
        plt.legend((self.flat_list(self.problem_sizes_string)))
        plt.savefig(plotname)

        return ax

    def plot_gate_counts_circuit_depth(self, plotname):
        data = read_json(self.filename)
        gate_counts_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext'])))
        avg_gate_counts = []

        x_axis = self.density
        # x_axis = [0.05, 0.1, 0.15]
        f, ax = plt.subplots()
        for i in range(0, len(self.problem_sizes_string)):
            for j in range(0, len(x_axis)):
                avg_gate_counts.append(sum(gate_counts_ext[i][j]))
            y_axis = avg_gate_counts
            avg_gate_counts = []
            ax.plot(x_axis, y_axis)
        plt.title('gates vs density')
        plt.xlabel('density')
        plt.ylabel('gate_counts')
        plt.legend((self.flat_list(self.problem_sizes_string)))
        plt.savefig(plotname)

        return ax

       # print(gate_counts_ext[0][12][0])  # problem size -  density - gate type(rx, ...)
       # print(gate_counts_ext[1][12][1])
       # print(gate_counts_ext[2][12][3])
       # print(gate_counts_ext[3][12][4])

    def countGates(self, plotname):
        data = read_json(self.filename)
        gate_counts_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext'])))

        ind = np.arange(4)
        width = 0.15

        fig = plt.subplots(figsize=(18, 12))
        prob3 = gate_counts_ext[0][12]
        bar1 = plt.bar(ind, prob3, width, color=self.colors[0], edgecolor='grey', label="N=3")
        prob4 = gate_counts_ext[1][12]
        bar2 = plt.bar(ind + width, prob4, width, color=self.colors[1], edgecolor='grey', label="N=4")
        prob5 = gate_counts_ext[2][12]
        bar3 = plt.bar(ind + width * 2, prob5, width, color=self.colors[2], edgecolor='grey', label="N=5")
        prob10 = gate_counts_ext[3][12]
        bar4 = plt.bar(ind + width * 3, prob10, width, color=self.colors[3], edgecolor='grey', label="N=10")

        plt.titel = "gate counts according to problem size "
        plt.xlabel('gates', fontweight='bold', fontsize=15)
        plt.ylabel('gate count', fontweight='bold', fontsize=15)
        plt.xticks([r + width for r in range(len(self.gate_keys))], self.gate_keys)

        plt.legend()
        plt.savefig(plotname)

    def gatesOverDensity(self, plotname):
        data = read_json(self.filename)
        gate_counts_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext'])))

        ind = np.arange(4)
        width = 0.07
        # problemsize
        p=0

        fig, ax = plt.subplots(figsize=(22, 14))

        den1 = gate_counts_ext[p][0]
        bar1 = ax.bar(ind, den1, width, color=self.colors[0], edgecolor='grey', label="d=0.05", alpha=.5)
        den2 = gate_counts_ext[p][1]
        bar2 = ax.bar(ind + width, den2, width, color=self.colors[1], edgecolor='grey', label="d=0.1", alpha=.5)
        den3 = gate_counts_ext[p][2]
        bar3 = ax.bar(ind + width*2, den3, width, color=self.colors[2], edgecolor='grey', label="d=0.15", alpha=.5)
        den4 = gate_counts_ext[p][3]
        bar4 = ax.bar(ind + width*3, den4, width, color=self.colors[3], edgecolor='grey', label="d=0.2", alpha=.5)
        den5 = gate_counts_ext[p][4]
        bar5 = ax.bar(ind + width*4, den5, width, color=self.colors[4], edgecolor='grey', label="d=0.25", alpha=.5)
        den6 = gate_counts_ext[p][5]
        bar6 = ax.bar(ind + width*5, den6, width, color=self.colors[5], edgecolor='grey', label="d=0.3", alpha=.5)
        den7 = gate_counts_ext[p][6]
        bar7 = ax.bar(ind + width * 6, den7, width, color=self.colors[6], edgecolor='grey', label="d=0.4", alpha=.5)
        den8 = gate_counts_ext[p][7]
        bar8 = ax.bar(ind + width * 7, den8, width, color=self.colors[7], edgecolor='grey', label="d=0.5", alpha=.5)
        den9 = gate_counts_ext[p][8]
        bar9 = ax.bar(ind + width * 8, den9, width, color=self.colors[8], edgecolor='grey', label="d=0.6", alpha=.5)
        den10 = gate_counts_ext[p][9]
        bar10 = ax.bar(ind+width*9, den10, width, color=self.colors[9], edgecolor='grey', label="d=0.7", alpha=.5)
        den11 = gate_counts_ext[p][10]
        bar11 = ax.bar(ind + width*10, den11, width, color=self.colors[10], edgecolor='grey', label="d=0.8", alpha=.5)
        den12 = gate_counts_ext[p][11]
        bar12 = ax.bar(ind + width*11, den12, width, color=self.colors[11], edgecolor='grey', label="d=0.9", alpha=.5)
        den13 = gate_counts_ext[p][12]
        bar13 = ax.bar(ind + width*12, den13, width, color=self.colors[12], edgecolor='grey', label="d=1.0", alpha=.5)

        ax.set_title("max3sat3_qubits6")
        plt.xlabel('gates', fontweight='bold', fontsize=15)
        plt.ylabel('gate count', fontweight='bold', fontsize=15)
        plt.xticks([r + width for r in range(len(self.gate_keys))], self.gate_keys)

        plt.legend()
        plt.savefig(plotname)

    def plot_density_circuit_depth_dot(self, problem, filename):
        lgs = 16
        ms = 10
        lw = 2
        mew = 2
        data = read_json(self.filename)
        depth_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['depth_ext'])))

        x_axis = self.density
        f, ax = plt.subplots(figsize=(20, 12))
        for i in range(0, len(self.problem_sizes_string)):
            y_axis = depth_ext[i]
            ax.errorbar(x_axis, y_axis, label='n = '+str(self.problem_sizes[i]), fmt='o--', markersize=ms, markeredgewidth=mew, markeredgecolor=self.dark_colors[i], markerfacecolor=self.colors[i], elinewidth=lw, ecolor=self.dark_colors[i], color=self.dark_colors[i])
        plt.title('circuit depth vs density ' + problem)
        plt.xlabel('density')
        plt.ylabel('circuit_depth')
        plt.xticks(x_axis)
        plt.legend(fontsize=lgs, ncol=2)


        #plt.show()
        plt.savefig(filename + ".png")

        return ax

    def plot_density_circuit_depth_dots_parallel(self, problem, filenames, filename):
        lgs = 16
        ms = 10
        lw = 2
        mew = 2

        depth_ext = []
        depth_ext_stds = []
        for k in range(len(filenames)):
            data = read_json(filenames[k])
            depth_ext.append(self.flat_list(self.flat_list(ast.literal_eval(data['results']['depth_ext'])[k])))
            depth_ext_stds.append(self.flat_list(self.flat_list(ast.literal_eval(data['results']['depth_ext_stds'])[k])))

        x_axis = self.density
        f, ax = plt.subplots(figsize=(20, 12))
        for i in range(0, len(self.problem_sizes_string)):
            y_axis = depth_ext[i]
            y_axis_err = depth_ext_stds[i]
            ax.errorbar(x_axis, np.log(y_axis), yerr=y_axis_err, label='n = '+str(self.problem_sizes[i]), fmt='o--', markersize=ms, markeredgewidth=mew, markeredgecolor=self.dark_colors[i], markerfacecolor=self.colors[i], elinewidth=lw, ecolor=self.dark_colors[i], color=self.dark_colors[i])
        plt.title('circuit depth vs density ' + problem)
        plt.xlabel('density')
        plt.ylabel('circuit_depth')
        ax.set_ylim([0, 6000])
        plt.xticks(x_axis)
        plt.legend(fontsize=lgs, ncol=2)


        plt.show()
        #plt.savefig(filename + ".png")

        return ax


    def plot_density_gate_counts_dots_parallel(self, problem, filenames, filename, gate, num):
        lgs = 16
        ms = 10
        lw = 2
        mew = 2

        gate_count_ext = []
        gate_count_std = []

        for i in range(len(filenames)):
            data = read_json(filenames[i])

            gate_count_ext.append(self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext'])[i])))
            gate_count_std.append(self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext_stds'])[i])))



        x_axis = self.density
        f, ax = plt.subplots(figsize=(20, 12))
        for i in range(0, len(self.problem_sizes_string)):
            y_axis_tmp = gate_count_ext[i]
            y_axis = [item[num] for item in y_axis_tmp]
            y_axis_err_tmp = gate_count_std[i]
            y_axis_err = [item[num] for item in y_axis_err_tmp]
            ax.errorbar(x_axis, y_axis, yerr=y_axis_err, label='n = '+str(self.problem_sizes[i]), fmt='o--', markersize=ms, markeredgewidth=mew, markeredgecolor=self.dark_colors[i], markerfacecolor=self.colors[i], elinewidth=lw, ecolor=self.dark_colors[i], color=self.dark_colors[i])
        plt.title('gate_count vs density ' + problem)
        plt.xlabel('density')
        plt.ylabel('gate_count_' + gate)
        plt.xticks(x_axis)
        ax.set_ylim([0, 32000])
        plt.legend(fontsize=lgs, ncol=2)


        #plt.show()
        plt.savefig(filename + ".png")

        return ax

    def plot_density_gate_counts_dots_parallel_swap(self, problem, filenames, filename):
        lgs = 16
        ms = 10
        lw = 2
        mew = 2

        swap_gates_mean = []
        swap_gates_std = []
        swap_counts = []
        ca = self.comp_averages

        for i in range(len(filenames)):
            data = read_json(filenames[i])
            swap_counts.append([self.flat_list(ast.literal_eval(data['results']['swap_gates']))])

        n = self.comp_averages

        swap = []
        for j in range(len(self.problem_sizes)):
            swap.append([swap_counts[j][0][i:i + n] for i in range(0, len(swap_counts[j][0]), n)])

        for l in range(len(swap)):
            for k in range(len(self.density)):
                swap_gates_mean.append(np.mean(swap[l][k]))
                swap_gates_std.append(np.std(swap[l][k]))


        x_axis = self.density
        f, ax = plt.subplots(figsize=(20, 12))
        for i in range(0, len(self.problem_sizes_string)):
            y_axis = swap_gates_mean[i*(len(self.density)):i*(len(self.density))+(len(self.density))]
            print(y_axis)
            y_axis_err = swap_gates_std[i*(len(self.density)):i*(len(self.density))+(len(self.density))]
            print(y_axis_err)
            ax.errorbar(x_axis, y_axis, yerr=y_axis_err, label='n = '+str(self.problem_sizes[i]), fmt='o--', markersize=ms, markeredgewidth=mew, markeredgecolor=self.dark_colors[i], markerfacecolor=self.colors[i], elinewidth=lw, ecolor=self.dark_colors[i], color=self.dark_colors[i])
        plt.title('swap_gates vs density ' + problem)
        plt.xlabel('density')
        plt.ylabel('gate_count_swap_gates')
        plt.xticks(x_axis)
        plt.legend(fontsize=lgs, ncol=2)


        #plt.show()
        plt.savefig(filename + ".png")

        return ax


    def plot_density_circuit_depth_dot2(self, problem, save_filename, filename2):
        lgs = 16
        ms = 10
        lw = 2
        mew = 2
        data = read_json(self.filename)
        depth_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['depth_ext'])))
        data2 = read_json(filename2)
        depth_ext2 = self.flat_list(self.flat_list(ast.literal_eval(data2['results']['depth_ext'])))

        x_axis = self.density
        f, ax = plt.subplots(figsize=(10, 7))
        for i in range(0, 3):
            y_axis = depth_ext[i]
            ax.errorbar(x_axis, y_axis, label='n = '+str(self.problem_sizes[i]), fmt='o--', markersize=ms, markeredgewidth=mew, markeredgecolor=self.dark_colors[i], markerfacecolor=self.colors[i], elinewidth=lw, ecolor=self.dark_colors[i], color=self.dark_colors[i])
        for j in range(0, 1):
            y_axis2 = depth_ext2[j]
            ax.errorbar(x_axis, y_axis2, label='n = '+str(self.problem_sizes[3+j]), fmt='o--', markersize=ms, markeredgewidth=mew, markeredgecolor=self.dark_colors[3+j], markerfacecolor=self.colors[3+j], elinewidth=lw, ecolor=self.dark_colors[3+j], color=self.dark_colors[3+j])
        plt.title('circuit depth vs density ' + problem)
        plt.xlabel('density')
        plt.ylabel('circuit_depth')
        plt.legend(fontsize=lgs, ncol=2)

        #plt.show()
        plt.savefig(save_filename + ".png")

        return ax

    def plot_gate_counts_circuit_depth_dot(self, plotname):
        data = read_json(self.filename)
        gate_counts_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext'])))
        avg_gate_counts = []

        x_axis = self.density
        # x_axis = [0.05, 0.1, 0.15]
        f, ax = plt.subplots()
        for i in range(0, len(self.problem_sizes_string)):
            for j in range(0, len(x_axis)):
                avg_gate_counts.append(sum(gate_counts_ext[i][j]))
            y_axis = avg_gate_counts
            avg_gate_counts = []
            ax.plot(x_axis, y_axis)
        plt.title('gates vs density')
        plt.xlabel('density')
        plt.ylabel('gate_counts')
        plt.legend((self.flat_list(self.problem_sizes_string)))
        plt.savefig(plotname)

        return ax



    def plot_gate_counts_circuit_depth_dot2(self, problem, save_filename, filename2):
        data = read_json(self.filename)
        gate_counts_ext = self.flat_list(self.flat_list(ast.literal_eval(data['results']['gate_counts_ext'])))
        avg_gate_counts = []
        data2 = read_json(filename2)
        depth_ext2 = self.flat_list(self.flat_list(ast.literal_eval(data2['results']['gate_counts_ext'])))
        avg_gate_counts2 = []

        x_axis = self.density
        # x_axis = [0.05, 0.1, 0.15]
        f, ax = plt.subplots()
        for i in range(0, len(self.problem_sizes_string)):
            for j in range(0, len(x_axis)):
                avg_gate_counts.append(sum(gate_counts_ext[i][j]))
            y_axis = avg_gate_counts
            avg_gate_counts = []
            ax.plot(x_axis, y_axis)
        plt.title('gates vs density')
        plt.xlabel('density')
        plt.ylabel('gate_counts')
        plt.legend((self.flat_list(self.problem_sizes_string)))
        plt.savefig(save_filename)

        return ax



    def combine_plots(self, list_plots):
        return list_plots



