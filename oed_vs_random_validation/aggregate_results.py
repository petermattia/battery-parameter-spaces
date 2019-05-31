import itertools
import numpy as np
import os, sys
import json
import pickle
import concurrent.futures

# real params
beta_list = [0.2,0.5,1.0,2.0,5.0,10.0,10000.0]
gamma_list = [0.00001, 0.1,0.3,1.0,3.0,10.0,30.0,100.0]
epsilon_list = [0.5,0.6,0.7,0.8,0.9,1.0]
num_hparams_configs = len(list(itertools.product(beta_list, gamma_list, epsilon_list)))

num_policies = 8
num_seeds = 100
pop_budget = 5
run_id = sys.argv[1]
logdir = './logs/' + run_id

policy_list = np.arange(num_policies).tolist()
max_budget = pop_budget*num_policies

script_dir = os.path.join(os.getcwd(), 'final_baseline', 'bash_scripts', run_id)
results_dict = [] # list of dicts of dicts

total_exp_folders = num_seeds*len(list(itertools.product(beta_list, gamma_list, epsilon_list)))
print(total_exp_folders, flush=True)

results_list = []
for _ in range(max_budget+1):
	results_list.append({})

def get_all_data(exp_id):
	if exp_id % 100 == 0:
		print(exp_id, flush=True)

	config_file = os.path.join(logdir, str(exp_id), 'config.json')

	with open(config_file) as infile:
		config = json.load(infile)

	assert (exp_id == int(config["exp_id"]))
	beta = config["init_beta"]
	gamma = config["gamma"]
	epsilon = config["epsilon"]
	seed = config["seed"]
	# train_policies = "".join(map(str,config["train_policy_idx"]))
	# if int(seed) != 0:
	# 	return None, None

	all_params = (seed, beta, gamma, epsilon)
	all_round_data = []
	for round_idx in range(max_budget+1):
		with open(os.path.join(logdir, str(exp_id), 'round_' + str(round_idx) +  '.txt')) as infile:
			lines = infile.readlines()
		true_lifetimes = [float(lifetime) for lifetime in lines[0].rstrip().split("\t")]
		pred_rankings = [int(rank) for rank in lines[2].rstrip().split("\t")]
		all_round_data.append((true_lifetimes, pred_rankings))
	return all_params, all_round_data

with concurrent.futures.ProcessPoolExecutor() as executor:
    for all_params, all_round_data in executor.map(get_all_data, range(1, total_exp_folders+1)):
    	if all_params is not None:
	        for around_idx, round_data in enumerate(all_round_data):
    		        results_list[around_idx][all_params] = round_data

with open(os.path.join(logdir, 'aggegated_results.pkl'), 'wb') as outfile:
	pickle.dump(results_list, outfile)

with open(os.path.join(logdir, 'aggegated_results.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)
	print(len(results_list))
