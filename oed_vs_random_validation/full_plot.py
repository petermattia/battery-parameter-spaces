import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, sys
from collections import defaultdict
import seaborn as sns
sns.set_style('white')

from scipy.stats import sem # standard errors
import itertools

np.random.seed(0)

num_policies = 9
pop_budget = 5
num_seeds = 2000
max_budget = num_policies*pop_budget
current_dir = 'oed_vs_random_validation'
data = np.genfromtxt(os.path.join(current_dir, 'final_results.csv'), delimiter=',', skip_header=1)
print(data)
sample_lifetimes = data[:, 3:3+pop_budget]
pop_lifetimes = np.mean(sample_lifetimes, axis=-1)
best_pop_lifetime = np.max(pop_lifetimes)
print(pop_lifetimes, np.mean(pop_lifetimes), np.std(pop_lifetimes), np.mean(np.std(sample_lifetimes, axis=-1)))

early_data = np.genfromtxt(os.path.join(current_dir, 'predictions.csv'), delimiter=',', skip_header=1)
print(early_data)

early_sample_lifetimes = early_data[:, 3:3+pop_budget]
early_sample_lifetimes = early_sample_lifetimes.tolist()
for policy_idx, policy_lifetimes in enumerate(early_sample_lifetimes):
	early_sample_lifetimes[policy_idx] = [lifetime for lifetime in policy_lifetimes if not np.isnan(lifetime)]


logdir_closed_loop = './logs/' + sys.argv[1]
logdir_oed = './logs/' + sys.argv[2]
# true_mean = 759.
true_mean = 772.3
biased_mean = 947.


####### Baseline 1: Random


random_lifetimes = np.zeros((num_seeds, max_budget+1))
random_best = np.zeros((num_seeds, max_budget+1))

for seed in range(num_seeds):
	if seed%100 == 0:
		print(seed, end=' ')
	random_policies = np.random.choice(num_policies, max_budget, replace=True)
	emp_policy_lifetimes = []
	for _ in range(num_policies):
		emp_policy_lifetimes.append([])

	for round_idx, policy in enumerate(random_policies):
		lifetime_cost = np.random.choice(sample_lifetimes[policy])
		emp_policy_lifetimes[policy].append(lifetime_cost)
		random_lifetimes[seed, round_idx+1] = lifetime_cost
		avg_emp_policy_lifetimes = []
		for policy_idx in range(num_policies):
			if len(emp_policy_lifetimes[policy_idx])>0:
				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
			else:
				avg_emp_policy_lifetimes.append(true_mean)
		random_best[seed, round_idx+1] = pop_lifetimes[np.argmax(np.array(avg_emp_policy_lifetimes))]

random_costs =  np.cumsum(random_lifetimes, axis=1)
random_cycles_means = np.mean(random_costs, axis=0)
random_cycles_stds = sem(random_costs, axis=0)

random_means = np.mean(random_best, axis=0)
random_stds = sem(random_best, axis=0)



random_guess = np.random.choice(pop_lifetimes, 2000)
random_means[0] = np.mean(random_guess)
random_stds[0] = sem(random_guess)



# random_means, random_stds, random_cycles_means, random_cycles_stds = [], [], [], []

# for round_idx in range(max_budget+1):
# 	print(round_idx, flush=True, end=' ')
# 	random_loss_list = []
# 	random_seed_cycles = []
# 	for seed in range(num_seeds):
# 		random_policies = np.random.choice(num_policies, round_idx+1, replace=True)
# 		emp_policy_lifetimes = []
# 		for _ in range(num_policies):
# 			emp_policy_lifetimes.append([])
# 		for random_policy in random_policies:
# 			emp_policy_lifetimes[random_policy].append(np.random.choice(sample_lifetimes[random_policy]))
# 		avg_emp_policy_lifetimes = []
# 		for policy_idx in range(num_policies):
# 			if len(emp_policy_lifetimes[policy_idx])>0:
# 				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
# 			else:
# 				avg_emp_policy_lifetimes.append(true_mean)
# 		random_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
# 		random_policy_lifetime = pop_lifetimes[random_policy]
# 		random_seed_cycles.append(random_policy_lifetime)
# 		# random_loss_list.append(best_pop_lifetime-random_policy_lifetime)
# 		random_loss_list.append(random_policy_lifetime)


# 	random_loss_np = np.array(random_loss_list)
# 	random_means.append(np.mean(random_loss_np))
# 	random_stds.append(sem(random_loss_np))

# 	random_seed_cycles_np = np.array(random_seed_cycles)
# 	random_cycles_means.append(np.mean(random_seed_cycles_np))
# 	random_cycles_stds.append(sem(random_seed_cycles_np))
print()
print(random_means)
print(random_stds)
print(random_cycles_means)
print(random_cycles_stds)
print()


####### Baseline 2: Grid Search

# grid_lifetimes = np.zeros((num_seeds, max_budget+1))
# grid_best = np.zeros((num_seeds, max_budget+1))

# for seed in range(num_seeds):
# 	if seed%100 == 0:
# 		print(seed, end=' ')
# 	ordered_policies = np.random.permutation([i for i in range(num_policies)]).tolist()*pop_budget
# 	emp_policy_lifetimes = []
# 	for _ in range(num_policies):
# 		emp_policy_lifetimes.append([])

# 	for round_idx, policy in enumerate(ordered_policies):
# 		lifetime_cost = np.random.choice(sample_lifetimes[policy])
# 		emp_policy_lifetimes[policy].append(lifetime_cost)
# 		grid_lifetimes[seed, round_idx+1] = lifetime_cost
# 		avg_emp_policy_lifetimes = []
# 		for policy_idx in range(num_policies):
# 			if len(emp_policy_lifetimes[policy_idx])>0:
# 				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
# 			else:
# 				avg_emp_policy_lifetimes.append(true_mean)
# 		grid_best[seed, round_idx+1] = pop_lifetimes[np.argmax(np.array(avg_emp_policy_lifetimes))]

# grid_costs =  np.cumsum(grid_lifetimes, axis=1)
# grid_cycles_means = np.mean(grid_costs, axis=0)
# grid_cycles_stds = sem(grid_costs, axis=0)

# grid_means = np.mean(grid_best, axis=0)
# grid_stds = sem(grid_best, axis=0)

# random_guess = np.random.choice(pop_lifetimes, 2000)
# grid_means[0] = np.mean(random_guess)
# grid_stds[0] = sem(random_guess)

# print()
# print(grid_means)
# print(grid_stds)
# print(grid_cycles_means)
# print(grid_cycles_stds)
# print()

# grid_means[0] = random_means[0]
# grid_stds[0] = random_stds[0]


####### Ablation Baseline 1: Early-pred + random

early_pred_lifetimes = np.zeros((num_seeds, max_budget+1))
early_pred_best = np.zeros((num_seeds, max_budget+1))

for seed in range(num_seeds):
	if seed%100 == 0:
		print(seed, end=' ')
	early_pred_policies = np.random.choice(num_policies, max_budget, replace=True)
	emp_policy_lifetimes = []
	for _ in range(num_policies):
		emp_policy_lifetimes.append([])

	for round_idx, policy in enumerate(early_pred_policies):
		lifetime_cost = np.random.choice(early_sample_lifetimes[policy])
		emp_policy_lifetimes[policy].append(lifetime_cost)
		early_pred_lifetimes[seed, round_idx+1] = lifetime_cost
		avg_emp_policy_lifetimes = []
		for policy_idx in range(num_policies):
			if len(emp_policy_lifetimes[policy_idx])>0:
				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
			else:
				avg_emp_policy_lifetimes.append(true_mean)
		early_pred_best[seed, round_idx+1] = pop_lifetimes[np.argmax(np.array(avg_emp_policy_lifetimes))]

early_pred_costs =  np.cumsum(early_pred_lifetimes, axis=1)
early_pred_cycles_means = np.mean(early_pred_costs, axis=0)
early_pred_cycles_stds = sem(early_pred_costs, axis=0)

early_pred_means = np.mean(early_pred_best, axis=0)
early_pred_stds = sem(early_pred_best, axis=0)

early_pred_means[0] = random_means[0]
early_pred_stds[0] = random_stds[0]

# early_pred_means, early_pred_stds, early_pred_cycles_means, early_pred_cycles_stds = [], [], [], []

# for round_idx in range(max_budget+1):
# 	print(round_idx, flush=True, end=' ')
# 	early_pred_loss_list = []
# 	early_pred_seed_cycles = []
# 	for seed in range(num_seeds):
# 		random_policies = np.random.choice(num_policies, round_idx+1, replace=True)
# 		emp_policy_lifetimes = []
# 		for _ in range(num_policies):
# 			emp_policy_lifetimes.append([])
# 		for random_policy in random_policies:
# 			emp_policy_lifetimes[random_policy].append(np.random.choice(early_sample_lifetimes[random_policy]))
# 		avg_emp_policy_lifetimes = []
# 		for policy_idx in range(num_policies):
# 			if len(emp_policy_lifetimes[policy_idx])>0:
# 				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
# 			else:
# 				avg_emp_policy_lifetimes.append(true_mean)
# 		random_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
# 		random_policy_lifetime = pop_lifetimes[random_policy]
# 		early_pred_seed_cycles.append(random_policy_lifetime)
# 		# early_pred_loss_list.append(best_pop_lifetime-random_policy_lifetime)
# 		early_pred_loss_list.append(random_policy_lifetime)


# 	early_pred_loss_np = np.array(early_pred_loss_list)
# 	early_pred_means.append(np.mean(early_pred_loss_np))
# 	early_pred_stds.append(sem(early_pred_loss_np))

# 	early_pred_seed_cycles_np = np.array(early_pred_seed_cycles)
# 	early_pred_cycles_means.append(np.mean(early_pred_seed_cycles_np))
# 	early_pred_cycles_stds.append(sem(early_pred_seed_cycles_np))
print()
print(early_pred_means)
print(early_pred_stds)
print(early_pred_cycles_means)
print(early_pred_cycles_stds)
print()

# early_pred_means[0] = grid_means[0]
# early_pred_stds[0] = grid_stds[0]

####### Ablation Baseline 2: OED w/o early_pred


oed_means, oed_stds = [random_means[0]], [random_stds[0]]
# with open(os.path.join(logdir_closed_loop, 'best_oed_losses.pkl'), 'rb') as infile:
with open(os.path.join(logdir_oed, 'best_oed_lifetimes.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)
	for all_round_data in results_list:
		all_round_data_np = np.array(all_round_data)
		oed_means.append(np.mean(all_round_data_np))
		oed_stds.append(sem(all_round_data_np))

oed_cycles_means, oed_cycles_stds = [0], [0]
with open(os.path.join(logdir_oed, 'best_experiment_cycles.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)
	for all_round_data in results_list:
		all_round_data_np = np.array(all_round_data)
		oed_cycles_means.append(np.mean(all_round_data_np))
		oed_cycles_stds.append(sem(all_round_data_np))

# # best hparams have beta 10000
oed_means[1] = random_means[1]
oed_stds[1] = random_stds[1]
oed_means[2] = random_means[2]
oed_stds[2] = random_stds[2]

####### Full closed loop

closed_loop_means, closed_loop_stds = [random_means[0]], [random_stds[0]]
# with open(os.path.join(logdir_closed_loop, 'best_oed_losses.pkl'), 'rb') as infile:
with open(os.path.join(logdir_closed_loop, 'best_oed_lifetimes.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)
	print(len(results_list))
	for all_round_data in results_list:
		all_round_data_np = np.array(all_round_data)
		closed_loop_means.append(np.mean(all_round_data_np))
		closed_loop_stds.append(sem(all_round_data_np))
closed_loop_means = np.array(closed_loop_means)
closed_loop_stds = np.array(closed_loop_stds)



plt.figure()
start_idx = 0
end_idx = start_idx+8
random_cycles_means[0] = 0
# grid_cycles_means[0] = 0
plt.xlim([-1000, 25000])
plt.plot([-1000, 25000], [890, 890], \
	'k-', \
	lw=2, \
	linestyle='--', \
	label='Optimal cycle life of best protocol')


plt.errorbar(np.cumsum(np.array(random_cycles_means))[start_idx:end_idx], \
	random_means[start_idx:end_idx], \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((random_stds[start_idx:end_idx], random_stds[start_idx:end_idx])), \
	xerr=1.96*np.vstack((random_cycles_stds[start_idx:end_idx], random_cycles_stds[start_idx:end_idx])), \
	marker='o', \
	linestyle=':', \
	label='Pure exploration, no early prediction')

# plt.errorbar(np.cumsum(np.array(grid_cycles_means))[start_idx:end_idx], \
# 	grid_means[start_idx:end_idx], \
# 	alpha=0.8, \
# 	linewidth=2, \
# 	yerr=1.96*np.vstack((grid_stds[start_idx:end_idx], grid_stds[start_idx:end_idx])), \
# 	xerr=1.96*np.vstack((grid_cycles_stds[start_idx:end_idx], grid_cycles_stds[start_idx:end_idx])), \
# 	marker='o', \
# 	linestyle=':', \
# 	label='grid baseline')




plt.errorbar(np.cumsum(np.array(oed_cycles_means))[start_idx:end_idx],  \
	oed_means[start_idx:end_idx], \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((oed_stds[start_idx:end_idx], oed_stds[start_idx:end_idx])), \
	xerr=1.96*np.vstack((oed_cycles_stds[start_idx:end_idx], oed_cycles_stds[start_idx:end_idx])), \
	marker='o', \
	linestyle=':', \
	label='Closed loop w/o early pred')

spaced_subset = np.array([0,1,2,3,4, 9, 14, 19, 24, 29, 34, 39, 44])
# spaced_subset = np.arange(max_budget)
print(early_pred_means[spaced_subset])
plt.errorbar(100*np.arange(start_idx,max_budget+1)[spaced_subset], 
	early_pred_means[spaced_subset], \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((early_pred_stds[spaced_subset], early_pred_stds[spaced_subset])), \
	marker='o', \
	linestyle=':', \
	label='Closed loop w/o oed')

plt.errorbar(100*np.arange(start_idx,max_budget+1)[spaced_subset], 
	closed_loop_means[spaced_subset], \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((closed_loop_stds[spaced_subset], closed_loop_stds[spaced_subset])), 
	marker='o', \
	linestyle=':', \
	label='Closed loop')

plt.ylabel('Average cycle life of current best protocol')
plt.xlabel('Experimental time (hours)')
# plt.xticks(np.arange(max_budget+1))
plt.legend()
plt.savefig(os.path.join(logdir_closed_loop, 'plot_full.png'))
plt.show()

all_data = {'no_oed_no_ep_x': np.cumsum(np.array(random_cycles_means))[start_idx:end_idx],
			'no_oed_no_ep_y': random_means[start_idx:end_idx],
			'no_oed_no_ep_yerr': np.vstack((random_stds[start_idx:end_idx], random_stds[start_idx:end_idx])),
			'no_oed_no_ep_xerr': np.vstack((random_cycles_stds[start_idx:end_idx], random_cycles_stds[start_idx:end_idx])),
			'oed_no_ep_x': np.cumsum(np.array(oed_cycles_means))[start_idx:end_idx],
			'oed_no_ep_y': oed_means[start_idx:end_idx],
			'oed_no_ep_yerr': np.vstack((oed_stds[start_idx:end_idx], oed_stds[start_idx:end_idx])),
			'oed_no_ep_xerr': np.vstack((oed_cycles_stds[start_idx:end_idx], oed_cycles_stds[start_idx:end_idx])),
			'no_oed_ep_x': 100*np.arange(start_idx,max_budget+1)[spaced_subset],
			'no_oed_ep_y': early_pred_means[spaced_subset],
			'no_oed_ep_yerr': np.vstack((early_pred_stds[spaced_subset], early_pred_stds[spaced_subset])),
			'oed_ep_x': 100*np.arange(start_idx,max_budget+1)[spaced_subset],
			'oed_ep_y': closed_loop_means[spaced_subset],
			'oed_ep_yerr': np.vstack((closed_loop_stds[spaced_subset], closed_loop_stds[spaced_subset]))
			}
print(closed_loop_means)
print(closed_loop_means[spaced_subset])
print(random_means)
print(np.cumsum(np.array(random_cycles_means))[start_idx:end_idx])
# all_data = {'random_x': np.cumsum(np.array(random_cycles_means))[start_idx:end_idx],
# 			'random_y': random_means[start_idx:end_idx],
# 			'random_yerr': np.vstack((random_stds[start_idx:end_idx], random_stds[start_idx:end_idx])),
# 			'random_xerr': np.vstack((random_cycles_stds[start_idx:end_idx], random_cycles_stds[start_idx:end_idx])),
# 			'grid_x': np.cumsum(np.array(grid_cycles_means))[start_idx:end_idx],
# 			'grid_y': grid_means[start_idx:end_idx],
# 			'grid_yerr': np.vstack((grid_stds[start_idx:end_idx], grid_stds[start_idx:end_idx])),
# 			'grid_xerr': np.vstack((grid_cycles_stds[start_idx:end_idx], grid_cycles_stds[start_idx:end_idx])),
# 			'oed_x': 100*np.arange(start_idx,max_budget+1),
# 			'oed_y': oed_means,
# 			'oed_yerr': np.vstack((oed_stds, oed_stds))}
with open(os.path.join(logdir_closed_loop, 'fig4_plot_data.pkl'), 'wb') as infile:
	pickle.dump(all_data, infile)



