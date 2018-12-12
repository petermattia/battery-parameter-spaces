import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data/testing/repeated.csv', delimiter=',')
print(data.shape)

policies = data[:, :3]
lifetimes = data[:, 3:6]
true_lifetimes = np.sum(lifetimes, axis=-1)/np.count_nonzero(lifetimes, axis=-1)
print(true_lifetimes)
np.random.seed(0)
num_seeds = 100
start_batch_size = 1
max_batch_size = 41
step_batch_size = 3
num_rounds = 3
num_policies = policies.shape[0]

best_random_lifetimes_mean = []
best_random_lifetimes_std = []

# Random with replacement

for batch_size in range(start_batch_size, max_batch_size, step_batch_size):
	best_seed_lifetimes = []
	for i in range(num_seeds):
		# np.random.seed(i)
		observed_lifetimes = np.zeros((num_policies, num_rounds))
		for rd in range(num_rounds):
			rd_policies = np.random.choice(num_policies, batch_size, replace=False)
			observed_lifetimes[np.ix_(rd_policies, [rd])] = lifetimes[np.ix_(rd_policies, [rd])]
		avg_lifetimes = np.sum(observed_lifetimes, axis=-1)/np.count_nonzero(observed_lifetimes, axis=-1)
		avg_lifetimes[np.isnan(avg_lifetimes)] = 0.
		best_seed_lifetimes.append(true_lifetimes[np.argmax(avg_lifetimes)])
	best_random_lifetimes_mean.append(np.mean(best_seed_lifetimes))
	best_random_lifetimes_std.append(np.std(best_seed_lifetimes))
best_random_lifetimes_mean = np.array(best_random_lifetimes_mean)
best_random_lifetimes_std = np.array(best_random_lifetimes_std)

print(len(best_random_lifetimes_mean), len(best_random_lifetimes_std))
print(list(range(start_batch_size, max_batch_size, step_batch_size)))
# plt.plot(list(range(start_batch_size, max_batch_size, step_batch_size)), best_random_lifetimes_mean)
plt.errorbar(list(range(start_batch_size, max_batch_size, step_batch_size)), best_random_lifetimes_mean, \
	linewidth=2, yerr=np.vstack((3*best_random_lifetimes_std, 3*best_random_lifetimes_std)), marker='o', \
	linestyle=':', label='random')
plt.ylabel('Averaged early predicted lifetime')
plt.xlabel('Number of arms per round')
plt.title('Total rounds = 3, seeds = 100')
plt.legend()
plt.savefig('random.png')
plt.show()
		
# Random without replacement

			
