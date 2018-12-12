"""
Script to generate predictions from simulator for closed_loop_oed simulations

Peter Attia
Last updated August 23, 2018
"""

import numpy as np
import argparse
import os

class GetRewards(object):

	def __init__(self, args):

		self.batch_file = os.path.join(args.data_dir, args.batch_dir, str(args.round_idx) + '.csv')
		self.pred_file  = os.path.join(args.data_dir, args.pred_dir,  str(args.round_idx) + '.csv') # note this is for previous round

		self.round_idx = args.round_idx

		pass

	def run(self):
		batch_file = self.batch_file
		pred_file = self.pred_file
		round_idx = self.round_idx

		# Read in new policies

		policies = np.genfromtxt(batch_file, delimiter=',')
		if policies.ndim == 1:
			policies = np.array([policies])

		data = np.genfromtxt('data/testing/repeated.csv', delimiter=',')
		all_policies = data[:, :3]
		all_lifetimes = data[:, 3:6]

		# Generate predictions
		pol_w_lifetimes = np.zeros((len(policies),5))
		for k, policy in enumerate(policies):
			C1 = policy[0]
			C2 = policy[1]
			C3 = policy[2]
			C4 = 0.2/(1/6 - (0.2/C1 + 0.2/C2 + 0.2/C3))
			policy_idx = np.where((all_policies == (C1, C2, C3)).all(axis=1))
			lifetime = all_lifetimes[policy_idx, round_idx]
			pol_w_lifetimes[k,:]= [C1,C2,C3,C4,lifetime]

		# Save predictions to pred/<round_idx>.csv
		np.savetxt(pred_file,pol_w_lifetimes,delimiter=',', fmt='%1.3f')


def parse_args():
	"""
	Specifies command line arguments for the program.
	"""

	parser = argparse.ArgumentParser(description='Retrieve predictions using recorded data')

	parser.add_argument('--data_dir', nargs='?', default='data_baseline/')
	parser.add_argument('--pred_dir', nargs='?', default='pred/')
	parser.add_argument('--batch_dir', nargs='?', default='batch/')
	parser.add_argument('--round_idx', default=0, type=int)

	return parser.parse_args()


def main():

	args = parse_args()

	np.set_printoptions(threshold=np.inf)

	assert (os.path.exists(os.path.join(args.data_dir, args.pred_dir)))
	assert (os.path.exists(os.path.join(args.data_dir, args.batch_dir)))

	agent = GetRewards(args)
	agent.run()

if __name__ == '__main__':

	main()
