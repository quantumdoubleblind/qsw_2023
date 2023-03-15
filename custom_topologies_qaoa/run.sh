#!/bin/sh
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



echo "Running the experiment for ${problem_to_solve} with graph density ${graph_densitiy} for ${comp_averages} runs in
${parallel_threads} parallel threads with ${count_layer} QAOA layer using a pptimization level of ${opt_level}"


python qsw_2023/custom_topologies_qaoa/src/experiments.py "$graph_densitiy" "$opt_level" "$comp_averages" "$parallel_threads" "$problem_to_solve" "$count_layer"
cmd /ks
