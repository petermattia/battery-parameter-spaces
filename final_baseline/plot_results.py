import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
import seaborn as sns
from scipy.stats import sem # standard errors
import itertools

num_policies = 6
pop_budget = 5
run_id = 'feb6'
logdir = './logs/' + run_id
policy_list = np.arange(num_policies).tolist()


with open(os.path.join(logdir, 'aggegated_results.pkl'), 'rb') as infile:
	results_list = pickle.load(infile)

full_policy_set = set([i for i in range(num_policies)])
def get_complement_policy(policy):
	policy_set = set([int(pidx) for pidx in policy])
	complement_set = np.array(list(full_policy_set.difference(policy_set)))
	complement_set.sort()
	return "".join(map(str, complement_set.tolist()))


max_budget = len(results_list)
oed_means = []
oed_stds = []
baseline_means = []
baseline_stds = []

for round_idx in range(max_budget):
	data = results_list[round_idx]
	print(round_idx)

	if round_idx == 4:
		break

	policy_loss_dict = defaultdict(list) # key: policy_list, hparam, value: list of losses
	best_policy_losses = []
	for exp_config, exp_result in data.items():
		best_policy_loss = np.amax(exp_result[0]) - exp_result[0][exp_result[1][0]]
		train_policy_set = np.array([int(pidx) for pidx in exp_config[1]])
		train_policy_set.sort()
		policy_config = ("".join(map(str, train_policy_set.tolist())), exp_config[2], exp_config[3], exp_config[4])
		
		# if (exp_config[2]==10000 and exp_config[3]==0.00001 and exp_config[4]==1.0):
		# 	print(policy_config, best_policy_loss)
		policy_loss_dict[policy_config].append(best_policy_loss)

	# print('Averages')
	avg_policy_loss_dict = {}
	for policy_config, policy_loss in policy_loss_dict.items():
		avg_policy_loss = np.mean(np.array(policy_loss))
		avg_policy_loss_dict[policy_config] = avg_policy_loss

		# if (policy_config[2]==10000 and policy_config[3]==0.00001 and policy_config[4]==1.0):
		# 	print(policy_config, avg_policy_loss)


	best_hparam_dict = defaultdict(lambda: (None, None, None, np.inf))
	for policy_config, policy_loss in avg_policy_loss_dict.items():
		if policy_loss < best_hparam_dict[policy_config[0]][3]:
			best_hparam_dict[policy_config[0]] = (policy_config[1], policy_config[2], policy_config[3], policy_loss)
	
	oed_loss_list = []
	baseline_loss_list = []
	for policy, best_hparam in best_hparam_dict.items():
		print(policy, best_hparam)
		complement_policy = get_complement_policy(policy)
		oed_loss = avg_policy_loss_dict[(complement_policy, best_hparam[0], best_hparam[1], best_hparam[2])]
		oed_loss_list.append(oed_loss)

		baseline_loss = avg_policy_loss_dict[(complement_policy, 10000, 0.00001, 1.0)]
		baseline_loss_list.append(baseline_loss)

		# print(policy, complement_policy, oed_loss_dict[policy])
	oed_loss_np = np.array(oed_loss_list)
	oed_means.append(np.mean(oed_loss_np))
	oed_stds.append(sem(oed_loss_np))

	baseline_loss_np = np.array(baseline_loss_list)
	baseline_means.append(np.mean(baseline_loss_np))
	baseline_stds.append(sem(baseline_loss_np))


print(oed_means)
print(oed_stds)

# print(baseline_means)
# print(baseline_stds)

baseline_means = []
baseline_stds = []

num_train_policies = 3
data = np.genfromtxt('final_baseline/predictions_3and3.csv', delimiter=',', skip_header=1)
print(data)
for round_idx in range(1, max_budget+1):
	print('Round', round_idx)
	baseline_loss_list = []
	for train_policy_list in itertools.combinations(policy_list, num_train_policies):
		
		train_data = data[list(train_policy_list)]
		sample_lifetimes = train_data[:, :3+pop_budget]
		pop_lifetimes = np.mean(sample_lifetimes, axis=-1)
		best_pop_lifetime = np.max(pop_lifetimes)

		for seed in range(100):
			emp_policy_lifetimes = []
			policies_selected = np.random.choice(num_train_policies, round_idx, replace=True).tolist()
			
			for _ in range(num_train_policies):
				emp_policy_lifetimes.append([])
			for policy_idx in policies_selected:
				emp_policy_lifetimes[policy_idx].append(np.random.choice(train_data[policy_idx]))

			avg_emp_policy_lifetimes = []
			for policy_idx in range(num_train_policies):
				if len(emp_policy_lifetimes[policy_idx]) > 0:
					avg_emp_policy_lifetimes.append(np.mean(np.array(emp_policy_lifetimes[policy_idx])))
				else:
					avg_emp_policy_lifetimes.append(-np.inf)
			random_policy = np.argmax(np.array(avg_emp_policy_lifetimes))
			baseline_loss_list.append(best_pop_lifetime-pop_lifetimes[random_policy])

	baseline_loss_np = np.array(baseline_loss_list)
	baseline_means.append(np.mean(baseline_loss_np))
	baseline_stds.append(sem(baseline_loss_np))

print(baseline_means)
print(baseline_stds)

plt.figure()
plt.errorbar(np.arange(max_budget)+1, baseline_means, alpha=0.8,\
	linewidth=2, yerr=np.vstack((baseline_stds, baseline_stds)), marker='o', \
	linestyle=':', label='random')
plt.errorbar(np.arange(max_budget)+1, oed_means, alpha=0.8, \
	linewidth=2, yerr=np.vstack((oed_stds, oed_stds)), marker='o', \
	linestyle=':', label='oed')
plt.ylabel('true best lifetime - predicted best lifetime')
plt.xlabel('Rounds')
plt.legend()
plt.savefig('final_baseline/summary_results/feb6.png')
plt.show()











