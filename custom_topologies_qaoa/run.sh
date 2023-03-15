#!/bin/sh
echo "What problem do you want to run: tsp, max3sat, maxcut or numpart"
read -r problem_to_solve
echo "Running the experiment for ${problem_to_solve}"
graph_densitiy="0.7"
count_layer="1"
opt_level="1"
comp_averages="3"
parallel_threads="2"
python experiments.py "$graph_densitiy" "$opt_level" "$comp_averages" "$parallel_threads" "$problem_to_solve" "$count_layer"
cmd /k