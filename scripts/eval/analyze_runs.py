import json
import os
import sys

from statistics import mean, stdev

root_dir = sys.argv[1]
commands = sys.argv[2:]

def compute_metrics(key, vals):
    values = [dct[key] for dct in vals]
    return (max(values), mean(values), stdev(values))

print("(max, mean, std. dev)")
for experiment in commands:
    exp_path = os.path.join(root_dir, experiment)
    if not os.path.exists(exp_path):
        print(f"Experiment not found: {experiment}")
        continue
    results_file = os.path.join(exp_path, 'results.jsonl')
    key_metrics = []
    with open(results_file, 'r') as raw_results:
        for line in raw_results:
            results_dict = json.loads(line)
            key_metrics.append({
                'best_epoch': results_dict['best_epoch'],
                'train_UAS': results_dict['training_UAS'],
                'train_LAS': results_dict['training_LAS'],
                'dev_UAS': results_dict['best_validation_UAS'],
                'dev_LAS': results_dict['best_validation_LAS'],
                'train_time': results_dict['training_duration']
            })
    print(f"{experiment}: {compute_metrics('dev_LAS', key_metrics)}") 

