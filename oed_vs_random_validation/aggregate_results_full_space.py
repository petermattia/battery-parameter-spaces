import itertools
import numpy as np
import os, sys
import json
import pickle
import concurrent.futures
from collections import defaultdict

# real params
beta_list = [0.2,0.5,1.0,2.0,5.0,10.0,10000.0]
gamma_list = [0.00001, 0.1,0.3,1.0,3.0,10.0,30.0,100.0]
epsilon_list = [0.5,0.6,0.7,0.8,0.9,1.0]


run_id = sys.argv[1]
logdir = './logs/' + run_id
num_seeds = 100


total_exp_folders = num_seeds*len(list(itertools.product(beta_list, gamma_list, epsilon_list)))
print(total_exp_folders, flush=True)

max_budget = 4

results_list = []
for _ in range(max_budget+1):
	results_list.append(defaultdict(list))

def get_all_data(exp_id):
	if exp_id % 100 == 0:
		print(exp_id,  end=' ', flush=True)

	config_file = os.path.join(logdir, str(exp_id), 'config.json')

	with open(config_file) as infile:
		config = json.load(infile)

	assert (exp_id == int(config["exp_id"]))
	beta = config["init_beta"]
	gamma = config["gamma"]
	epsilon = config["epsilon"]
	seed = config["seed"]


	all_params = (seed, beta, gamma, epsilon)
	all_round_data = []
	for round_idx in range(max_budget+1):
		with open(os.path.join(logdir, str(exp_id), 'round_' + str(round_idx) +  '.txt')) as infile:
			lines = infile.readlines()
		true_lifetimes = np.array([float(lifetime) for lifetime in lines[0].rstrip().split("\t")])
		pred_rankings = [int(rank) for rank in lines[2].rstrip().split("\t")]
		oed_best = true_lifetimes[pred_rankings][0]
		all_round_data.append(oed_best)
	return all_params, all_round_data

# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for all_params, all_round_data in executor.map(get_all_data, range(1, total_exp_folders+1)):
#     	if all_params is not None:
# 	        for round_idx, round_data in enumerate(all_round_data):
#     		    # print(round_idx, all_params, round_data)
#     		    results_list[round_idx][(all_params[1], all_params[2], all_params[3])].append(round_data)


# with open(os.path.join(logdir, 'aggregated_results_full_space.pkl'), 'wb') as outfile:
# 	pickle.dump(results_list, outfile)

with open(os.path.join(logdir, 'aggregated_results_full_space.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)
	print(len(results_list))

print('aggregate dumped...')

max_perfs = []
max_perfs_all_seeds = []
for round_idx in range(max_budget+1):
	print(round_idx, end=' ', flush=True)
	round_max_perf = -np.inf
	round_max_perf_all_seeds = []
	for hparam_config, hparam_values in results_list[round_idx].items():
		avg_hparam_perf = np.mean(hparam_values)
		if avg_hparam_perf > round_max_perf:
			round_max_perf = avg_hparam_perf
			round_max_perf_all_seeds = hparam_values
	max_perfs.append(round_max_perf)
	max_perfs_all_seeds.append(round_max_perf_all_seeds)
print(max_perfs)
with open(os.path.join(logdir, 'best_results_full_space.pkl'), 'wb') as outfile:
	pickle.dump((max_perfs, max_perfs_all_seeds), outfile)

with open(os.path.join(logdir, 'best_results_full_space.pkl'), 'rb') as infile:
	max_perfs, max_perfs_all_seeds = pickle.load(infile)
	print(len(max_perfs), len(max_perfs_all_seeds))

