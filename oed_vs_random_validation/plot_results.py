import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, sys
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import ListedColormap
sns.set_style('white')
my_cmap = ListedColormap(sns.color_palette('colorblind'))
color=iter(my_cmap(np.arange(10)))
from scipy.stats import sem # standard errors
import itertools

np.random.seed(0)

num_policies = 8
pop_budget = 5
max_budget = num_policies*pop_budget
current_dir = 'oed_vs_random_validation'
data = np.genfromtxt(os.path.join(current_dir, 'final_results.csv'), delimiter=',', skip_header=1)
print(data)
sample_lifetimes = data[:, 3:3+pop_budget]
pop_lifetimes = np.mean(sample_lifetimes, axis=-1)
best_pop_lifetime = np.max(pop_lifetimes)

run_id = sys.argv[1]
logdir = './logs/' + run_id
rd_mean = 759

baseline_means, baseline_stds, baseline_cycles_means, baseline_cycles_stds = [], [], [], []

for round_idx in range(max_budget+1):
	print(round_idx, flush=True, end=' ')
	baseline_loss_list = []
	baseline_seed_cycles = []
	for seed in range(2000):
		random_policies = np.random.choice(num_policies, round_idx+1, replace=True)
		emp_policy_lifetimes = []
		for _ in range(num_policies):
			emp_policy_lifetimes.append([])
		for random_policy in random_policies:
			emp_policy_lifetimes[random_policy].append(np.random.choice(sample_lifetimes[random_policy]))
		avg_emp_policy_lifetimes = []
		for policy_idx in range(num_policies):
			if len(emp_policy_lifetimes[policy_idx])>0:
				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
			else:
				avg_emp_policy_lifetimes.append(rd_mean)
		random_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
		random_policy_lifetime = pop_lifetimes[random_policy]
		baseline_seed_cycles.append(random_policy_lifetime)
		# baseline_loss_list.append(best_pop_lifetime-random_policy_lifetime)
		baseline_loss_list.append(random_policy_lifetime)


	baseline_loss_np = np.array(baseline_loss_list)
	baseline_means.append(np.mean(baseline_loss_np))
	baseline_stds.append(sem(baseline_loss_np))

	baseline_seed_cycles_np = np.array(baseline_seed_cycles)
	baseline_cycles_means.append(np.mean(baseline_seed_cycles_np))
	baseline_cycles_stds.append(sem(baseline_seed_cycles_np))
print()
print(baseline_means)
print(baseline_stds)
print(baseline_cycles_means)
print(baseline_cycles_stds)

grid_means, grid_stds, grid_cycles_means, grid_cycles_stds = [], [], [], []

for round_idx in range(max_budget+1):
	print(round_idx, flush=True, end=' ')
	grid_loss_list = []
	grid_seed_cycles = []
	for seed in range(2000):
		ordered_policies = np.random.permutation([i for i in range(num_policies)]).tolist()*pop_budget
		grid_policies = ordered_policies[:round_idx+1]
		emp_policy_lifetimes = []
		for _ in range(num_policies):
			emp_policy_lifetimes.append([])
		for grid_policy in grid_policies:
			emp_policy_lifetimes[grid_policy].append(np.random.choice(sample_lifetimes[grid_policy]))
		avg_emp_policy_lifetimes = []
		for policy_idx in range(num_policies):
			if len(emp_policy_lifetimes[policy_idx])>0:
				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
			else:
				avg_emp_policy_lifetimes.append(rd_mean)
		grid_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
		grid_policy_lifetime = pop_lifetimes[grid_policy]
		grid_seed_cycles.append(grid_policy_lifetime)
		# grid_loss_list.append(best_pop_lifetime-grid_policy_lifetime)
		grid_loss_list.append(grid_policy_lifetime)


	grid_loss_np = np.array(grid_loss_list)
	grid_means.append(np.mean(grid_loss_np))
	grid_stds.append(sem(grid_loss_np))

	grid_seed_cycles_np = np.array(grid_seed_cycles)
	grid_cycles_means.append(np.mean(grid_seed_cycles_np))
	grid_cycles_stds.append(sem(grid_seed_cycles_np))
print()
print(grid_means)
print(grid_stds)
print(grid_cycles_means)
print(grid_cycles_stds)

oed_means, oed_stds = [grid_means[0]], [grid_stds[0]]
# with open(os.path.join(logdir, 'best_oed_losses.pkl'), 'rb') as infile:
with open(os.path.join(logdir, 'best_oed_lifetimes.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)
	for all_round_data in results_list:
		all_round_data_np = np.array(all_round_data)
		oed_means.append(np.mean(all_round_data_np))
		oed_stds.append(sem(all_round_data_np))


plt.figure()
start_idx = 0
end_idx = start_idx+6
baseline_cycles_means[0]=0
grid_cycles_means[0] = 0
baseline_means[0] = grid_means[0]
baseline_stds[0] = grid_stds[0]
c = next(color)
plt.errorbar(np.cumsum(np.array(baseline_cycles_means))[start_idx:end_idx], \
	baseline_means[start_idx:end_idx], \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((baseline_stds[start_idx:end_idx], baseline_stds[start_idx:end_idx])), \
	xerr=1.96*np.vstack((baseline_cycles_stds[start_idx:end_idx], baseline_cycles_stds[start_idx:end_idx])), \
	marker='o', \
	linestyle=':', \
	color=c, \
	# color='r', \
	label='random')
c = next(color)
plt.errorbar(np.cumsum(np.array(grid_cycles_means))[start_idx:end_idx], \
	grid_means[start_idx:end_idx], \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((grid_stds[start_idx:end_idx], grid_stds[start_idx:end_idx])), \
	xerr=1.96*np.vstack((grid_cycles_stds[start_idx:end_idx], grid_cycles_stds[start_idx:end_idx])), \
	marker='o', \
	linestyle=':', \
	color=c, \
	# color='y', \
	label='grid')
c = next(color)
plt.errorbar(100*np.arange(start_idx,max_budget+1), 
	oed_means, \
	alpha=0.8, \
	linewidth=2, \
	yerr=1.96*np.vstack((oed_stds, oed_stds)), 
	marker='o', \
	linestyle=':', \
	color=c, \
	# color='b', \
	label='closed loop')
plt.ylabel('cycle life of best policy found so far')
plt.xlabel('experimentation time (in battery cycles)')
# plt.xticks(np.arange(max_budget+1))
plt.legend()
plt.savefig(os.path.join(logdir, 'plot_full.png'))
plt.show()


# all_data = {'random_x': np.cumsum(np.array(baseline_cycles_means))[start_idx:end_idx],
# 			'random_y': baseline_means[start_idx:end_idx],
# 			'random_yerr': np.vstack((baseline_stds[start_idx:end_idx], baseline_stds[start_idx:end_idx])),
# 			'random_xerr': np.vstack((baseline_cycles_stds[start_idx:end_idx], baseline_cycles_stds[start_idx:end_idx])),
# 			'grid_x': np.cumsum(np.array(grid_cycles_means))[start_idx:end_idx],
# 			'grid_y': grid_means[start_idx:end_idx],
# 			'grid_yerr': np.vstack((grid_stds[start_idx:end_idx], grid_stds[start_idx:end_idx])),
# 			'grid_xerr': np.vstack((grid_cycles_stds[start_idx:end_idx], grid_cycles_stds[start_idx:end_idx])),
# 			'oed_x': 100*np.arange(start_idx,max_budget+1),
# 			'oed_y': oed_means,
# 			'oed_yerr': np.vstack((oed_stds, oed_stds))}
# with open(os.path.join(logdir, 'fig4_plot_data.pkl'), 'wb') as infile:
# 	pickle.dump(all_data, infile)



