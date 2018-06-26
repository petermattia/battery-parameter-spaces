import numpy as np
import argparse
from sim4step import sim

class BayesGap(object):

	def __init__(self, args):

		self.param_space = self.get_parameter_space()
		self.num_arms = self.param_space.shape[0]

		self.X = self.get_design_matrix(args.kernel_bandwidth)

		self.sim_mode = args.sim_mode
		self.num_dims = self.X.shape[1]
		self.batch_size = args.bsize
		self.budget = args.budget # T in Alg. 1

		self.eta = args.prior_std
		self.sigma = args.likelihood_std
		self.beta = args.beta
		self.epsilon = args.epsilon

		pass

	def get_design_matrix(self, kernel_bandwidth):

		from sklearn.kernel_approximation import (RBFSampler, Nystroem)
		param_space = self.param_space
		num_arms = self.num_arms

		feature_map_nystroem = Nystroem(gamma=kernel_bandwidth, n_components=num_arms, random_state=1)
		X = feature_map_nystroem.fit_transform(param_space)
		# print('start', X[:3])
		# print('end', X[-3:])

		"""
		def Gaussian(x1,x2):
			return np.exp((-(np.linalg.norm(x1-x2)**2))/(2*kernel_bandwidth**2))
			
		kernel_matrix = np.zeros((num_arms, num_arms))

		for i in range(num_arms):
			kernel_matrix[i, i] = 1.
			for j in range(i+1, num_arms):
				kernel_matrix[i, j] = Gaussian(param_space[i], param_space[j])
				kernel_matrix[j, i] = kernel_matrix[i, j]

		


		
		# # bad approximation based on eigenvalues
		from numpy import linalg as LA
		w, v = LA.eigh(kernel_matrix)
		X = np.dot(v.T, np.diag(np.sqrt(w)))
		for i in range(num_arms):
			X[i, :] = np.sqrt(w[i]) * v[:, i]

		print('start', X[:3])
		print('end', X[-3:])
		# # alt that works well
		
		from scipy.linalg import sqrtm
		X = sqrtm(kernel_matrix)
		print('start sqrt', X[:3])
		print('end sqrt', X[-3:])
		"""

		return X


	def run(self):
		"""
		Algorithm 1 of paper
		"""

		num_arms = self.num_arms
		budget = self.budget
		batch_size = self.batch_size
		epsilon = self.epsilon
		X = self.X

		def find_J_t(carms):

			B_k_ts = []
			# print('n', num_arms)
			for k in range(num_arms):
				# print('k', k)
				if k in carms:
					temp_upper_bounds = np.delete(upper_bounds, k)
					B_k_t = np.amax(temp_upper_bounds) 
					B_k_ts.append(B_k_t)
					# print('B_k_t:', B_k_t, len(temp_upper_bounds), np.argmax(temp_upper_bounds))
					# print(temp_upper_bounds)
				else:
					B_k_ts.append(np.inf)
					# print('B_k_t:inf')
			np.set_printoptions(precision=2)
			# print(np.array(upper_bounds))
			# print(np.array(B_k_ts))
			# exit()
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

		best_arm = 0

		X_t = []
		Y_t = []
		proposal_arms = [] # useful for Eq (8) later
		proposal_gaps = []
		upper_bounds, lower_bounds = self.get_posterior_bounds()

		for round_idx in range(budget):
			print('Round', round_idx)
			batch_arms = []
			candidate_arms = list(range(num_arms)) # an extension of Alg 1 to batch setting, don't select the arm again in same batch
			for batch_elem in range(batch_size):
				J_t, _ = find_J_t(candidate_arms)
				j_t = find_j_t(candidate_arms, J_t)
				s_J_t = get_confidence_diameter(J_t)
				s_j_t = get_confidence_diameter(j_t)
				a_t = J_t if s_J_t >= s_j_t else j_t

				# print('a:', a_t, 'J:', J_t, 'j:', j_t, 'bool', s_J_t >= s_j_t)
				# print('s_J:%f, s_j: %f' % (s_J_t, s_j_t))

				if batch_elem == 0:
					proposal_arms.append(J_t)
					# proposal_gaps.append(proposal_gap)
				batch_arms.append(a_t)
				candidate_arms.remove(a_t)
				


			print('Policy indices', batch_arms)
			rewards = self.observe_reward(batch_arms)
			self.beta = self.beta * epsilon
			X_t.append(X[batch_arms])
			Y_t.append(rewards)
			upper_bounds, lower_bounds = self.get_posterior_bounds(np.vstack(X_t), np.vstack(Y_t))

			temp_upper_bounds = np.delete(upper_bounds, proposal_arms[round_idx])
			B_k_t = np.amax(temp_upper_bounds) - lower_bounds[proposal_arms[round_idx]]
			proposal_gaps.append(B_k_t)
			print()

		print(proposal_arms)
		print(proposal_gaps)
		best_arm = proposal_arms[np.argmin(np.array(proposal_gaps))]
		return self.param_space[best_arm]

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
		z=np.dot(posterior_covar, prior_mean) / (eta * eta)
		posterior_mean = np.linalg.multi_dot((posterior_covar, X_t.T, Y_t))/ (sigma * sigma) 
		# + np.dot(posterior_covar, np.reshape(prior_mean, [-1, 1])) / (eta * eta)
		

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

	def get_posterior_bounds(self, X=None, Y=None):
		"""
		Returns upper and lower bounds for all arms at every time step.
		"""


		"""
		Ignore block of code below

		num_arms = self.num_arms

		def get_beta(self):

			# Eq. (9) in paper


			def get_delta_k(self, k):

				# Heuristic after Theorem 1 in the paper


				lower_reference = marginal_mean[k] - 3 * marginal_var[k]
				delta_k = np.max(upper_references-lower_reference)

				return delta_k

			upper_references = marginal_mean + 3 * marginal_var

			sqinv_H_k_epsilon = []
			for _ in range(num_arms):
				delta_k = self.get_delta_k(marginal_mu_params)
				H_k_epsilon = max(0.5 * (delta_k + epsilon), epsilon)
				sqinv_H_k_epsilon.append(1/(H_k_epsilon*H_k_epsilon))

			H_epsilon = sum(sqinv_H_k_epsilon)

		"""
		beta = self.beta

		posterior_theta_params = self.posterior_theta(X, Y)
		marginal_mu_params = self.marginal_mu(posterior_theta_params)
		marginal_mean, marginal_var = marginal_mu_params

		upper_bounds = marginal_mean + beta * marginal_var
		lower_bounds = marginal_mean - beta * marginal_var

		return (upper_bounds, lower_bounds)


	def get_parameter_space(self):

		policies = np.genfromtxt('data/policies_all.csv',
				delimiter=',', skip_header=0)
		print(policies.shape)
		return policies[:, :3]

	def observe_reward(self, selected_arms):

		param_space = self.param_space
		rewards = []

		for arm in selected_arms:
			params = param_space[arm]
			reward = sim(params[0], params[1], params[2], mode=self.sim_mode, variance=True)
			rewards.append(reward)
			print(params, reward)

		rewards = np.array(rewards).reshape((-1,1))
		return rewards

def parse_args():
	"""
	Specifies command line arguments for the program.
	"""

	parser = argparse.ArgumentParser(description='Best arm identification using Bayes Gap.')
	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--budget', default=10, type=int,
						help='Time budget')
	parser.add_argument('--bsize', default=48, type=int,
						help='batch size')
	parser.add_argument('--datadir', nargs='?', default='data/',
						help='Directory for cycling data')
	parser.add_argument('--prior_std', default=100, type=float,
						help='standard deviation for the prior')
	parser.add_argument('--kernel_bandwidth', default=0.2, type=float,
						help='kernel bandwidth for Gaussian kernel')
	parser.add_argument('--likelihood_std', default=2.19, type=float,
						help='standard deviation for the likelihood std')
	parser.add_argument('--beta', default=1, type=float,
						help='exploration constant in Thm 1')
	parser.add_argument('--epsilon', default=1, type=float,
						help='decay constant for exploration')

	parser.add_argument('--sim_mode', nargs='?', default='lo')

	return parser.parse_args()


def main():

	args = parse_args()
	# args_dict = vars(args)
	# globals().update(args_dict)

	np.random.seed(args.seed)
	np.set_printoptions(threshold=np.inf)

	agent = BayesGap(args)
	best_arm_params = agent.run()
	print('Best arm:', best_arm_params)
	print('Lifetime:', sim(best_arm_params[0], best_arm_params[1], best_arm_params[2], mode=args.sim_mode, variance=False))


if __name__ == '__main__':

	main()
