import json
import os
import sys

from statistics import mean, stdev

PREFIX = 'Raw LAS: '

root_dir = sys.argv[1]
commands = sys.argv[2:]

def compute_metrics(values):
    return (max(values), mean(values), stdev(values))

print("(max, mean, std. dev)")
for experiment in commands:
    exp_path = os.path.join(root_dir, experiment)
    if not os.path.exists(exp_path):
        print(f"Experiment not found: {experiment}")
        continue
    results_file = os.path.join(exp_path, 'test_report.txt')
    las_scores = []
    with open(results_file, 'r') as raw_results:
        for line in raw_results:
            if PREFIX in line:
                las_scores.append(float(line[line.index(PREFIX) + len(PREFIX):]))
    print(f"{experiment}: {compute_metrics(las_scores)}") 

