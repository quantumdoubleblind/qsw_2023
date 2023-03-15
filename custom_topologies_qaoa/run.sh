#!/bin/sh
problem_to_solve="tsp"
graph_densitiy="0.7"
count_layer="1"
opt_level="1"
comp_averages="3"
parallel_threads="2"

echo "What problem do you want to run: tsp, max3sat, maxcut or numpart"
read -r problem_to_solve

echo "Choose your graph_densitiy: Number between 0.0 and 1.0"
read -r graph_densitiy

echo "Choose your optimization level: 1, 2 or 3"
read -r opt_level

echo "How many runs per density"
read -r comp_averages

echo "How many layers do you need in your QAOA algorithm"
read -r count_layer

echo "How many parallel_threads can you run"
read -r parallel_threads



echo "Running the experiment for ${problem_to_solve}"


python qsw_2023/custom_topologies_qaoa/src/experiments.py "$graph_densitiy" "$opt_level" "$comp_averages" "$parallel_threads" "$problem_to_solve" "$count_layer"
cmd /ks
