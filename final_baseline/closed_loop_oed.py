import numpy as np
import argparse
import pickle
import os
import csv

class BayesGap(object):

	def __init__(self, args):

		self.policy_file = args.policy_file
		self.train_policy_idx = args.train_policy_idx
		# np.random.shuffle(self.train_policy_idx) 
		self.pop_budget = args.pop_budget

		self.prev_arm_bounds_file = os.path.join(args.logdir, args.exp_id, args.arm_bounds_dir, str(args.round_idx-1) + '.pkl') # note this is for previous round
		self.arm_bounds_file = os.path.join(args.logdir, args.exp_id, args.arm_bounds_dir, str(args.round_idx) + '.pkl')
		self.next_batch_file = os.path.join(args.logdir, args.exp_id, args.next_batch_dir, str(args.round_idx) + '.csv')
		self.early_pred = args.early_pred

		if args.early_pred:
			self.prev_early_pred_file = os.path.join(args.logdir, args.exp_id, args.early_pred_dir, str(args.round_idx-1) + '.csv') # note this is for previous round
		else:
			self.prev_batch_lifetimes_file = os.path.join(args.logdir, args.exp_id, args.sampled_lifetimes_dir, str(args.round_idx-1) + '.csv')
			self.next_batch_lifetimes_file = os.path.join(args.logdir, args.exp_id, args.sampled_lifetimes_dir, str(args.round_idx) + '.csv')

		self.param_space = self.get_parameter_space()
		self.num_arms = self.param_space.shape[0]

		self.X = self.get_design_matrix(args.gamma)

		self.num_dims = self.X.shape[1]
		self.batch_size = args.bsize
		self.round_idx = args.round_idx

		self.sigma = args.likelihood_std
		self.beta = args.init_beta
		self.epsilon = args.epsilon

		self.standardization_mean = args.standardization_mean
		self.standardization_std = args.standardization_std

		self.eta = self.standardization_std

		pass

	def get_design_matrix(self, gamma):

		from sklearn.kernel_approximation import (RBFSampler, Nystroem)
		param_space = self.param_space
		num_arms = self.num_arms

		feature_map_nystroem = Nystroem(gamma=gamma, n_components=num_arms, random_state=1)
		X = feature_map_nystroem.fit_transform(param_space)
		return X

	def run(self):
		"""
		Algorithm 1 of paper
		"""

		prev_arm_bounds_file = self.prev_arm_bounds_file
		arm_bounds_file = self.arm_bounds_file
		next_batch_file = self.next_batch_file

		if self.early_pred:
			prev_early_pred_file = self.prev_early_pred_file
		else:
			prev_batch_lifetimes_file = self.prev_batch_lifetimes_file
			next_batch_lifetimes_file = self.next_batch_lifetimes_file

		num_arms = self.num_arms
		batch_size = self.batch_size
		epsilon = self.epsilon
		X = self.X
		round_idx = self.round_idx
		param_space = self.param_space

		def find_J_t(carms):

			B_k_ts = []
			for k in range(num_arms):
				if k in carms:
					temp_upper_bounds = np.delete(upper_bounds, k)
					B_k_t = np.amax(temp_upper_bounds)
					B_k_ts.append(B_k_t)
				else:
					B_k_ts.append(np.inf)

			B_k_ts = np.array(B_k_ts) - np.array(lower_bounds)
			J_t = np.argmin(B_k_ts)
			min_B_k_t = np.amin(B_k_ts)
			return J_t, min_B_k_t


		def find_j_t(carms, preselected_arm):

			U_k_ts = []
			for k in range(num_arms):
				if k in carms and k != preselected_arm:
					U_k_ts.append(upper_bounds[k])
				else:
					U_k_ts.append(-np.inf)

			j_t = np.argmax(np.array(U_k_ts))

			return j_t


		def get_confidence_diameter(k):

			return upper_bounds[k] - lower_bounds[k]


		if round_idx == 0:
			X_t = []
			Y_t = []
			proposal_arms = [] # useful for Eq (8) later
			proposal_gaps = []
			beta = self.beta
			upper_bounds, lower_bounds = self.get_posterior_bounds(beta)
			best_arm_params = None
			rank_idx = None
		else:

			# load proposal_arms, proposal_gaps, X_t, Y_t, beta for previous round in bounds/<round_idx-1>.pkl
			with open(prev_arm_bounds_file, 'rb') as infile:
				proposal_arms, proposal_gaps, X_t, Y_t, beta = pickle.load(infile)

			# update beta for this round
			beta = np.around(beta * epsilon, 4)

			# get armidx of batch policies and early predictions for previous round in pred/<round_idx-1>.csv

			if self.early_pred:
				with open(prev_early_pred_file, 'r', encoding='utf-8-sig') as infile:
					reader = csv.reader(infile, delimiter=',')
					early_pred = np.asarray([list(map(float, row)) for row in reader])
				print('Early predictions')
				print(early_pred)
				print()
				print('Standardized early predictions')
				early_pred[:, -1] = early_pred[:, -1] - self.standardization_mean
				print(early_pred)
				print()

				batch_policies = early_pred[:, :3]
				batch_arms = [param_space.tolist().index(policy) for policy in batch_policies.tolist()]
				batch_rewards = early_pred[:, 4].reshape(-1, 1) # this corresponds to 5th column coz we are supposed to ignore the 4th column
			else:
				with open(prev_batch_lifetimes_file) as infile:
					reader = csv.reader(infile, delimiter=',')
					resampled_lifetimes = np.asarray([list(map(float, row)) for row in reader])
				print('Resampled policy + non-standardized lifetime')
				print(resampled_lifetimes)
				resampled_lifetimes[:, -1] = resampled_lifetimes[:, -1] - self.standardization_mean
				

				batch_policies = resampled_lifetimes[:, :3]
				batch_arms = [param_space.tolist().index(policy) for policy in batch_policies.tolist()]
				batch_rewards = resampled_lifetimes[:, 3].reshape(-1, 1) # this corresponds to 4th column coz we dumped 4th column as lifetime here
			
			X_t.append(X[batch_arms])
			Y_t.append(batch_rewards)

			np_X_t = np.vstack(X_t)
			np_Y_t = np.vstack(Y_t)
			upper_bounds, lower_bounds = self.get_posterior_bounds(beta, np_X_t, np_Y_t)
			# J_prev_round = proposal_arms[round_idx-1]
			# temp_upper_bounds = np.delete(upper_bounds, J_prev_round)
			# B_k_t = np.amax(temp_upper_bounds) - lower_bounds[J_prev_round]
			# proposal_gaps.append(B_k_t)
			# best_arm = proposal_arms[np.argmin(np.array(proposal_gaps))]
			# best_arm_params = param_space[best_arm]

			# just take best upper bound arm
			best_arm_idx = np.argmax(upper_bounds)
			best_arm_params = param_space[best_arm_idx]
			rank_idx = np.argsort(-upper_bounds) # sort in descending order
			

		print('Arms with (non-standardized) upper bounds, lower bounds, and mean (upper+lower)/2 lifetimes')
		nonstd_upper_bounds = upper_bounds+self.standardization_mean
		nonstd_lower_bounds = lower_bounds+self.standardization_mean
		for ((policy_id, policy_param), ub, lb, mean) in zip(enumerate(param_space), nonstd_upper_bounds, nonstd_lower_bounds, (nonstd_upper_bounds+nonstd_lower_bounds)/2):
			print(policy_id, policy_param, ub, lb, mean, sep='\t')
		# Save bounds 
		with open(arm_bounds_file[:-4]+'_bounds.pkl', 'wb') as outfile:
			pickle.dump([param_space, nonstd_upper_bounds, nonstd_lower_bounds, (nonstd_upper_bounds+nonstd_lower_bounds)/2], outfile)
		print()

		print('Round', round_idx)
		print('Current beta', beta)
		batch_arms = []
		candidate_arms = list(range(num_arms)) # an extension of Alg 1 to batch setting, don't select the arm again in same batch
		for batch_elem in range(batch_size):
			J_t, _ = find_J_t(candidate_arms)
			j_t = find_j_t(candidate_arms, J_t)
			s_J_t = get_confidence_diameter(J_t)
			s_j_t = get_confidence_diameter(j_t)
			a_t = J_t if s_J_t >= s_j_t else j_t

			if batch_elem == 0:
				proposal_arms.append(J_t)
			batch_arms.append(a_t)
			candidate_arms.remove(a_t)

		# print('Policy indices selected for this round:', batch_arms)

		# save proposal_arms, proposal_gaps, X_t, Y_t, beta for current round in bounds/<round_idx>.pkl
		with open(arm_bounds_file, 'wb') as outfile:
			pickle.dump([proposal_arms, proposal_gaps, X_t, Y_t, beta], outfile)

		# save policies corresponding to batch_arms in batch/<round_idx>.csv
		batch_policies = [param_space[arm] for arm in batch_arms]
		with open(next_batch_file, 'w') as outfile:
			writer = csv.writer(outfile)
			writer.writerows(batch_policies)

		if not self.early_pred: # write sampled lifetimes for next round
			with open(next_batch_lifetimes_file, 'w+') as outfile:
				writer = csv.writer(outfile)
				for arm, policy in zip(batch_arms, batch_policies):
					resampled_lifetime = np.random.choice(self.sampled_lifetimes[arm])[None]
					print('Next round selection is arm idx', arm, ' with policy params', policy)
					# print('Resampled lifetime', resampled_lifetime)
					writer.writerow(np.concatenate((policy, resampled_lifetime)))
		print()
		return best_arm_params, rank_idx

	def posterior_theta(self, X_t, Y_t):

		"""
		Eq. (4, 5)
		"""

		num_dims = self.num_dims
		sigma = self.sigma
		eta = self.eta
		prior_mean = np.zeros(num_dims)

		prior_theta_params = (prior_mean, eta * eta * np.identity(num_dims))

		if X_t is None:
			return prior_theta_params

		posterior_covar = np.linalg.inv(np.dot(X_t.T, X_t) / (sigma * sigma) + np.identity(num_dims) / (eta * eta))
		posterior_mean = np.linalg.multi_dot((posterior_covar, X_t.T, Y_t))/ (sigma * sigma)

		posterior_theta_params = (np.squeeze(posterior_mean), posterior_covar)
		return posterior_theta_params


	def marginal_mu(self, posterior_theta_params):

		# marginal_mu denoted by rho_t

		# Note expected reward for each arm, mu_k = E[Y|theta] is another random variable

		X = self.X
		posterior_mean, posterior_covar = posterior_theta_params

		marginal_mean = np.dot(X, posterior_mean) # dimensions num_arms x num_dims
		marginal_var = np.sum(np.multiply(np.dot(X, posterior_covar), X), 1)
		marginal_mu_params = (marginal_mean, marginal_var)

		return marginal_mu_params

	def get_posterior_bounds(self, beta, X=None, Y=None):
		"""
		Returns upper and lower bounds for all arms at every time step.
		"""

		posterior_theta_params = self.posterior_theta(X, Y)
		marginal_mu_params = self.marginal_mu(posterior_theta_params)
		marginal_mean, marginal_var = marginal_mu_params

		upper_bounds = marginal_mean + beta * np.sqrt(marginal_var)
		lower_bounds = marginal_mean - beta * np.sqrt(marginal_var)

		upper_bounds = np.around(upper_bounds, 4)
		lower_bounds = np.around(lower_bounds, 4)

		return (upper_bounds, lower_bounds)


	def get_parameter_space(self):

		data = np.genfromtxt(self.policy_file, delimiter=',')
		
		data = data[self.train_policy_idx, :]
		
		policies = data[:, :3]
		
		self.sampled_lifetimes = data[:, 3:3+self.pop_budget]
		self.population_lifetimes = np.sum(self.sampled_lifetimes, axis=-1)/np.count_nonzero(self.sampled_lifetimes, axis=-1)
		
		print('Ordered policy list', self.train_policy_idx)
		print('Full data')
		print(data)
		print()
		return policies

def parse_args():
	"""
	Specifies command line arguments for the program.
	"""

	parser = argparse.ArgumentParser(description='Best arm identification using Bayes Gap.')

	parser.add_argument('--policy_file', nargs='?', default='data/testing/high_grad.csv')
	parser.add_argument('--arm_bounds_dir', nargs='?', default='bounds/')
	parser.add_argument('--early_pred_dir', nargs='?', default='pred/')
	parser.add_argument('--next_batch_dir', nargs='?', default='batch/')
	parser.add_argument('--round_idx', default=0, type=int)

	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--bsize', default=1, type=int,
						help='batch size')

	parser.add_argument('--gamma', default=1, type=float,
						help='kernel bandwidth for Gaussian kernel')
	parser.add_argument('--likelihood_std', default=164, type=float,
						help='standard deviation for the likelihood std')
	parser.add_argument('--init_beta', default=5.0, type=float,
						help='initial exploration constant in Thm 1')
	parser.add_argument('--epsilon', default=0.5, type=float,
						help='decay constant for exploration')

	parser.add_argument('--standardization_mean', default=947.0, type=float,
						help='mean lifetime from batch8')
	parser.add_argument('--standardization_std', default=164, type=float,
						help='std lifetime from batch8')

	# new args
	parser.add_argument('--logdir', nargs='?', default='logs/')
	parser.add_argument('--exp_id', default='0', type=str,
						help='unique experiment id')
	parser.add_argument('--train_policy_idx', nargs='*', default=[0,1,5,7], type=int,
						help='list of train policies')
	parser.add_argument('--pop_budget', default=3, type=int,
						help='Number of samples for computing population budget')
	parser.add_argument('--early_pred', dest='early_pred', action='store_true', default=False,
                        help='uses early prediction if true')
	parser.add_argument('--max-budget', default=20, type=int,
						help='maximum number of rounds to run experiment')
	parser.add_argument('--dump', dest='dump', action='store_true', default=False,
                        help='dumps to log file if true')
	parser.add_argument('--sampled_lifetimes_dir', nargs='?', default='sample/')


	return parser.parse_args()


def main():

	args = parse_args()

	np.random.seed(args.seed+1000*args.round_idx)
	np.set_printoptions(threshold=np.inf)

	resdir = os.path.join(args.logdir, args.exp_id)
	log_file = os.path.join(resdir, 'round_' + str(args.round_idx) + '.txt')	

	if ((os.path.exists(resdir) and args.round_idx != 0) or (os.path.isfile(log_file))) \
		and args.exp_id != '0': # exp_id = 0 is reserved for testing code and can be rewritten
		print('Results folder', resdir, os.path.exists(resdir))
		print('Log file', log_file, os.path.isfile(log_file))
		print('Results folder/log file already exists...exiting')
		exit()

	if args.round_idx == 0:

		if args.exp_id != '0':
			os.mkdir(resdir)
			os.mkdir(os.path.join(resdir, args.arm_bounds_dir))
			os.mkdir(os.path.join(resdir, args.next_batch_dir))
			if args.early_pred:
				os.mkdir(os.path.join(resdir, args.early_pred_dir))
				raise NotImplementedError('STILL NEED TO MAKE SURE W/ and W/O EARLY PRED FILES\
				 FOR bounds/ batch/ logs.csv DONT CONFLICT. TECHNICALLY OKA')
			else:
				os.mkdir(os.path.join(resdir, args.sampled_lifetimes_dir))


		import json
		with open(os.path.join(resdir, 'config.json'), 'w') as fp:
		    json.dump(vars(args), fp, indent=4, separators=(',', ': '))

	if args.dump:
		import sys
		sys.stdout = open(os.path.join(resdir, 'round_' + str(args.round_idx) + '_log.txt'), 'a+')

	agent = BayesGap(args)
	best_arm_params, rank_idx = agent.run()

	if args.round_idx != 0:
		print('Best arm until round', args.round_idx-1, \
			'is', best_arm_params, \
			'with population lifetime', agent.population_lifetimes[rank_idx[0]])

		# Log the best arm at the end of each round
		with open(log_file, 'w') as outfile:
			outfile.write('\t'.join(map(str, agent.population_lifetimes))+'\n')
			outfile.write('\t'.join(map(str, np.argsort(-agent.population_lifetimes)))+'\n')
			outfile.write('\t'.join(map(str, rank_idx))+'\n')
			outfile.write('\t'.join(map(str, best_arm_params)))

			print('True lifetimes:', '\t'.join(map(str, agent.population_lifetimes)))
			print('True ranks:', '\t'.join(map(str, np.argsort(-agent.population_lifetimes))))
			print('Predicted ranks:', '\t'.join(map(str, rank_idx)))
			print('Predicted best arm parameters:', '\t'.join(map(str, best_arm_params)))

	# delete pickle files to save memory
	if args.round_idx == args.max_budget:
		import shutil
		shutil.rmtree(os.path.join(resdir, args.arm_bounds_dir))
		shutil.rmtree(os.path.join(resdir, args.next_batch_dir))
		if args.early_pred:
				shutil.rmtree(os.path.join(resdir, args.early_pred_dir))
		else:
			shutil.rmtree(os.path.join(resdir, args.sampled_lifetimes_dir))



if __name__ == '__main__':

	main()
