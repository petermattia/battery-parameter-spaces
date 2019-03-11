import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
import seaborn as sns
from scipy.stats import sem # standard errors
import itertools

np.random.seed(0)

num_policies = 8
pop_budget = 5
max_budget = num_policies*pop_budget

data = np.genfromtxt('paper_plot/final_results.csv', delimiter=',', skip_header=1)
print(data)
sample_lifetimes = data[:, 3:3+pop_budget]
pop_lifetimes = np.mean(sample_lifetimes, axis=-1)
best_pop_lifetime = np.max(pop_lifetimes)


baseline_means, baseline_stds = [], []

for round_idx in range(max_budget):
	print(round_idx, flush=True, end=' ')
	baseline_loss_list = []
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
				avg_emp_policy_lifetimes.append(-np.inf)
		random_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
		random_policy_lifetime = pop_lifetimes[random_policy]
		baseline_loss_list.append(best_pop_lifetime-random_policy_lifetime)

	baseline_loss_np = np.array(baseline_loss_list)
	baseline_means.append(np.mean(baseline_loss_np))
	baseline_stds.append(sem(baseline_loss_np))

print(baseline_means)
print(baseline_stds)

grid_means, grid_stds = [], []
all_policies = [i for i in range(num_policies)]*pop_budget
for round_idx in range(max_budget):
	print(round_idx, flush=True, end=' ')
	grid_loss_list = []

	for seed in range(2000):
		grid_policies = np.random.permutation(all_policies)[:round_idx+1]
		emp_policy_lifetimes = []
		for _ in range(num_policies):
			emp_policy_lifetimes.append([])
		for grid_policy in grid_policies:
			finished_trials = len(emp_policy_lifetimes[grid_policy])
			emp_policy_lifetimes[grid_policy].append(sample_lifetimes[grid_policy][finished_trials])
		avg_emp_policy_lifetimes = []
		for policy_idx in range(num_policies):
			if len(emp_policy_lifetimes[policy_idx])>0:
				avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
			else:
				avg_emp_policy_lifetimes.append(-np.inf)
		grid_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
		grid_policy_lifetime = pop_lifetimes[grid_policy]
		grid_loss_list.append(best_pop_lifetime-grid_policy_lifetime)

	grid_loss_np = np.array(grid_loss_list)
	grid_means.append(np.mean(grid_loss_np))
	grid_stds.append(sem(grid_loss_np))

print(grid_means)
print(grid_stds)

plt.figure()
start_idx = 0
plt.errorbar(np.arange(1,max_budget+1), baseline_means[start_idx:], alpha=0.8,\
	linewidth=2, yerr=np.vstack((baseline_stds[start_idx:], baseline_stds[start_idx:])), marker='o', \
	linestyle=':', label='random')
plt.errorbar(np.arange(1,max_budget+1), grid_means[start_idx:], alpha=0.8,\
	linewidth=2, yerr=np.vstack((grid_stds[start_idx:], grid_stds[start_idx:])), marker='o', \
	linestyle=':', label='grid')
# plt.errorbar(np.arange(start_idx,max_budget+1), oed_means[start_idx:], alpha=0.8, \
# 	linewidth=2, yerr=np.vstack((oed_stds[start_idx:], oed_stds[start_idx:])), marker='o', \
# 	linestyle=':', label='oed')
plt.ylabel('true best lifetime - predicted best lifetime')
plt.xlabel('budget')
plt.xticks(np.arange(max_budget+1))
plt.legend()
plt.savefig('paper_plot/summary_results/mar10.png')
plt.show()

