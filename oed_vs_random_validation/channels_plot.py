import numpy as np
import matplotlib.pyplot as plt
from sim_with_seed_validation import *
import os
import sys
import pickle
from scipy.stats import sem 

num_channels = [1, 8, 16, 24, 48]
num_rounds = [1, 2, 3, 4]
num_seeds = 100

np.random.seed(0)
current_dir = 'oed_vs_random_validation'
data = np.genfromtxt(os.path.join(current_dir, 'policies_all.csv'), delimiter=',', skip_header=1)
policies = data[:, :3]
num_policies = len(policies)

# simulate random baseline

random_performance_means = []
random_performance_stds = []
for channel in num_channels:
	channel_performance = []
	for seed in range(num_seeds):
		seed_perfomance = []
		np.random.seed(seed)
		lifetimes = np.zeros(num_policies)
		non_zero = np.zeros(num_policies)
		# print('SEED', seed)
		for round_idx in num_rounds:
			test_policy_idx = np.random.choice(num_policies, channel, replace=False)
			non_zero[test_policy_idx] += 1
			test_policies = policies[test_policy_idx]
			# print(test_policy_idx, end=' ', flush=True)
			sim_seed = seed + 1000*round_idx
			test_lifetimes = [sim(test_policy[0], test_policy[1], test_policy[2], 
							variance=True, seed=sim_seed, early_pred=False, apply_correction=False) \
							for test_policy in test_policies]

			lifetimes[test_policy_idx] = (lifetimes[test_policy_idx]*(non_zero[test_policy_idx]-1) + test_lifetimes)/non_zero[test_policy_idx]

			best_lifetime_policy = policies[np.argmax(lifetimes)]
			# print(best_lifetime_policy, np.argmax(lifetimes), np.max(lifetimes))
			seed_perfomance.append(sim(best_lifetime_policy[0], best_lifetime_policy[1], best_lifetime_policy[2], 
							variance=False, seed=sim_seed, early_pred=False, apply_correction=False))
		# 	print(lifetimes)
		# 	print()
		# if seed > 5:
		# 	exit()
		channel_performance.append(seed_perfomance)
	channel_performance = np.array(channel_performance)
	print(channel_performance.shape)
	print(np.mean(channel_performance, axis=0))
	random_performance_means.append(np.mean(channel_performance, axis=0))
	random_performance_stds.append(sem(channel_performance, axis=0))

print(random_performance_means, random_performance_stds)

with open(os.path.join('logs', 'random_results_full_space.pkl'), 'wb') as outfile:
	pickle.dump((random_performance_means, random_performance_stds), outfile)

with open(os.path.join('logs', 'random_results_full_space.pkl'), 'rb') as infile:
	random_performance = pickle.load(infile)
	print(random_performance)

# with open(os.path.join('logs', 'sim_ablation.pkl'), 'rb') as infile:
# 	random_performance_means, random_performance_stds, oed_performance_means, oed_performance_stds = pickle.load(infile)


run_files = []
oed_performance_means = []
oed_performance_stds = []

for i in range(len(num_channels)):
	run_files.append(sys.argv[i+1])
	with open(os.path.join('logs', run_files[i], 'best_results_full_space.pkl'), 'rb') as infile:
		_, max_perfs_all_seeds = pickle.load(infile)
	channel_performance = np.array(max_perfs_all_seeds).T
	print(channel_performance.shape)
	oed_performance_means.append(np.mean(channel_performance, axis=0))
	oed_performance_stds.append(sem(channel_performance, axis=0))

for i, channels in enumerate(num_channels):
	plt.errorbar(np.array(num_rounds), 
		random_performance_means[i], \
		alpha=0.8, \
		linewidth=2, \
		yerr=1.96*np.vstack((random_performance_stds[i], random_performance_stds[i])), 
		marker='o', \
		linestyle=':', \
		label='Pure exploration, no early prediction')

	plt.errorbar(np.array(num_rounds), 
		oed_performance_means[i][1:], \
		alpha=0.8, \
		linewidth=2, \
		yerr=1.96*np.vstack((oed_performance_stds[i][1:], oed_performance_stds[i][1:])), 
		marker='o', \
		linestyle=':', \
		label='Closed loop (w/o early prediction)')

	plt.ylabel('Average cycle life of current best protocol')
	plt.xlabel('Number of rounds of testing')
	# plt.xticks(np.arange(max_budget+1))
	plt.legend()
	plt.title('number of channels: ' + str(channels))
	plt.savefig(os.path.join('logs', 'num_channels_' +str(channels)))
	plt.show()

with open(os.path.join('logs', 'sim_ablation.pkl'), 'wb') as outfile:
	pickle.dump((random_performance_means, random_performance_stds, \
		oed_performance_means, oed_performance_stds), outfile)












