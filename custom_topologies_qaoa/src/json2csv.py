# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

import csv
from python2json import read_json
import ast
import numpy as np

ps = [3, 4, 6, 9, 16, 25, 36, 49, 64, 81, 100]
#ps = [6, 9, 16, 25, 36, 49, 64, 81, 100]
#ps = [3, 4, 5, 6, 13, 22, 33, 46, 61, 78, 97]
d = [0.013895, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
             0.9, 1.0]

def flat_list(l):
    return [item for sublist in l for item in sublist]


def csv_converter(filename, save_file):
    data = read_json(filename)
    comp_averages = 20

    times = ast.literal_eval(data['results']['times'])
    depths = ast.literal_eval(data['results']['depths'])
    gate_counts = ast.literal_eval(data['results']['gate_counts'])
    swap_counts = ast.literal_eval(data['results']['swap_count'])
    times_ext = flat_list(flat_list(ast.literal_eval(data['results']['times_ext'])))
    times_ext_stds = flat_list(flat_list(ast.literal_eval(data['results']['times_ext_stds'])))
    depth_ext = flat_list(flat_list(ast.literal_eval(data['results']['depth_ext'])))
    depth_ext_stds = flat_list(flat_list(ast.literal_eval(data['results']['depth_ext_stds'])))
    gate_counts_ext = flat_list(flat_list(ast.literal_eval(data['results']['gate_counts_ext'])))
    gate_counts_ext_stds = flat_list(flat_list(ast.literal_eval(data['results']['gate_counts_ext_stds'])))

    with open(save_file + '.csv', "w", newline="") as outfile:
        counter = -1
        writer = csv.writer(outfile)
        writer.writerow(['num_qubits', 'density', 'circuit_depth', 'circuit_depth_mean', 'circuit_depth_std',
                         'compilaton_time', 'compilation_time_mean', 'compilation_time_std', 'gate_counts_rz',
                         'gate_counts_sx', 'gate_counts_x', 'gate_counts_cx'
                                                            'gate_counts_rz_std', 'gate_counts_sx_std',
                         'gate_counts_x_std', 'gate_counts_cx_std',
                         'gate_counts_rz_mean', 'gate_counts_sx_mean', 'gate_counts_x_mean', 'gate_counts_cx_mean',
                         'swap_count'])
        for n in range(len(ps)):
            for cd in range(len(d)):
                counter = counter + 1
                for i in range(comp_averages):
                    print(counter * 20)
                    print(counter * 20 + 20)
                    print(swap_counts[counter * 20:counter * 20 + 20])
                    writer.writerow(
                        [ps[int(n)], d[cd], depths[counter * 20][i], depth_ext[n][cd], depth_ext_stds[n][cd],
                         times[counter * 20][i], times_ext[n][cd], times_ext_stds[n][cd],
                         gate_counts[counter * 20][i][0], gate_counts[counter * 20][i][1],
                         gate_counts[counter * 20][i][2], gate_counts[counter * 20][i][3],
                         gate_counts_ext_stds[n][cd][0], gate_counts_ext_stds[n][cd][1],
                         gate_counts_ext_stds[n][cd][2], gate_counts_ext_stds[n][cd][3],
                         gate_counts_ext[n][cd][0], gate_counts_ext[n][cd][1], gate_counts_ext[n][cd][2],
                         gate_counts_ext[n][cd][3], swap_counts[counter * 20:counter * 20 + 20][i][0]])
        outfile.close()


def csv_converter_list_of_files(filenames, save_file, save_raw):
    times = []
    depths = []
    gate_counts = []
    swap_counts = []
    times_ext = []
    times_ext_stds = []
    depth_ext = []
    depth_ext_stds = []
    gate_counts_ext = []
    gate_counts_ext_stds = []

    comp_averages = 20

    for i in range(len(filenames)):
        data = read_json(filenames[i])
        times.append(ast.literal_eval(data['results']['times']))
        depths.append(ast.literal_eval(data['results']['depths']))
        gate_counts.append(ast.literal_eval(data['results']['gate_counts']))
        swap_counts.append(ast.literal_eval(data['results']['swap_gates']))
        times_ext.append(flat_list(flat_list(ast.literal_eval(data['results']['times_ext'])[i])))
        times_ext_stds.append(flat_list(flat_list(ast.literal_eval(data['results']['times_ext_stds'])[i])))
        depth_ext.append(flat_list(flat_list(ast.literal_eval(data['results']['depth_ext'])[i])))
        depth_ext_stds.append(flat_list(flat_list(ast.literal_eval(data['results']['depth_ext_stds'])[i])))
        gate_counts_ext.append(flat_list(flat_list(ast.literal_eval(data['results']['gate_counts_ext'])[i])))
        gate_counts_ext_stds.append(flat_list(flat_list(ast.literal_eval(data['results']['gate_counts_ext_stds'])[i])))

    with open(save_file + '.csv', "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['num_qubits', 'density', 'circuit_depth_mean', 'circuit_depth_std',
                         'compilation_time_mean', 'compilation_time_std',
                         'gate_counts_rz_std', 'gate_counts_sx_std', 'gate_counts_x_std', 'gate_counts_cx_std',
                         'gate_counts_rz_mean', 'gate_counts_sx_mean', 'gate_counts_x_mean', 'gate_counts_cx_mean'])
        for n in range(len(ps)):
            for cd in range(len(d)):
                writer.writerow([ps[int(n)], d[cd], depth_ext[n][cd],
                                 depth_ext_stds[n][cd], times_ext[n][cd],
                                 times_ext_stds[n][cd], gate_counts_ext_stds[n][cd][0],
                                 gate_counts_ext_stds[n][cd][1], gate_counts_ext_stds[n][cd][2],
                                 gate_counts_ext_stds[n][cd][3], gate_counts_ext[n][cd][0],
                                 gate_counts_ext[n][cd][1], gate_counts_ext[n][cd][2], gate_counts_ext[n][cd][3]])
    outfile.close()

    with open(save_raw + '.csv', "w", newline="") as outfile2:
        writer = csv.writer(outfile2)
        writer.writerow(['num_qubits', 'density', 'circuit_depth',
                         'compilaton_time', 'gate_counts_rz',
                         'gate_counts_sx', 'gate_counts_x', 'gate_counts_cx', 'swap_gates'])
        for n in range(len(ps)):
            for cd in range(len(d)):
                for j in range(0, comp_averages):
                    writer.writerow([ps[int(n)],
                                     d[cd],
                                     depths[n][cd][j],
                                     times[n][cd][j],
                                     gate_counts[n][cd][j][0],
                                     gate_counts[n][cd][j][1],
                                     gate_counts[n][cd][j][2],
                                     gate_counts[n][cd][j][3],
                                     swap_counts[n][cd][0]])
    outfile2.close()


def csv_converter_list_of_files_raw(filenames, save_file):
    times = []
    depths = []
    gate_counts = []
    swap_counts = []

    comp_averages = 20

    for i in range(len(filenames)):
        data = read_json(filenames[i])
        times.append(ast.literal_eval(data['results']['times']))
        depths.append(ast.literal_eval(data['results']['depths']))
        gate_counts.append(ast.literal_eval(data['results']['gate_counts']))
        swap_counts.append(ast.literal_eval(data['results']['swap_gates']))

    # print(depths)

    with open(save_file + '.csv', "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['num_qubits', 'density', 'circuit_depth',
                         'compilaton_time', 'gate_counts_rz',
                         'gate_counts_sx', 'gate_counts_x', 'gate_counts_cx', 'swap_counts'])
        for n in range(len(ps)):
            for cd in range(int((len(depths[0])) / comp_averages)):
                print(cd)
                for j in range(0, comp_averages):
                    writer.writerow([ps[int(n)],
                                     d[cd],
                                     depths[n][cd * 20][j],
                                     times[n][cd * 20][j],
                                     gate_counts[n][cd * 20][j][0],
                                     gate_counts[n][cd * 20][j][1],
                                     gate_counts[n][cd * 20][j][2],
                                     gate_counts[n][cd * 20][j][3],
                                     swap_counts[n][cd * 20][0]])
        outfile.close()


