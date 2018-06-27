"""
Script to generate predictions from simulator for closed_loop_oed simulations

Peter Attia
Last updated June 26, 2018
"""

import numpy as np
import argparse
from sim_with_seed import sim
import os

class GetRewards(object):

	def __init__(self, args):

		self.batch_file = os.path.join(args.data_dir, args.batch_dir, str(args.round_idx) + '.csv')
		self.pred_file  = os.path.join(args.data_dir, args.pred_dir,  str(args.round_idx) + '.csv') # note this is for previous round

		self.sim_mode = args.sim_mode
		self.round_idx = args.round_idx
		self.seed = args.seed

		pass

	def run(self):
         batch_file = self.batch_file
         pred_file = self.pred_file
         sim_mode = self.sim_mode
         seed = self.seed
        
         # Read in new policies
         policies = np.genfromtxt(batch_file, delimiter=',')
         
         # Generate predictions
         pol_w_lifetimes = np.zeros((len(policies),5))
         for k, policy in enumerate(policies):
             C1 = policy[0]
             C2 = policy[1]
             C3 = policy[2]
             C4 = 0.2/(1/6 - (0.2/C1 + 0.2/C2 + 0.2/C3))
             lifetime = sim(C1, C2, C3, sim_mode,seed=seed)
             pol_w_lifetimes[k,:]= [C1,C2,C3,C4,lifetime]
             
         # Remove one or two policies
         if np.random.random()>0.5:
             pol_w_lifetimes = np.delete(pol_w_lifetimes, (-1), axis=0)
             
         if np.random.random()>0.5:
             pol_w_lifetimes = np.delete(pol_w_lifetimes, (-1), axis=0)
         
         # Save predictions to pred/<round_idx>.csv
         np.savetxt(pred_file,pol_w_lifetimes,delimiter=',', fmt='%1.3f')


def parse_args():
	"""
	Specifies command line arguments for the program.
	"""

	parser = argparse.ArgumentParser(description='Generate predictions using thermal simulator')

	parser.add_argument('--data_dir', nargs='?', default='data2/')
	parser.add_argument('--pred_dir', nargs='?', default='pred/')
	parser.add_argument('--batch_dir', nargs='?', default='batch/')
	parser.add_argument('--round_idx', default=0, type=int)

	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--sim_mode', nargs='?', default='lo')

	return parser.parse_args()


def main():

	args = parse_args()

	np.random.seed(args.seed)
	np.set_printoptions(threshold=np.inf)

	assert (os.path.exists(os.path.join(args.data_dir, args.pred_dir)))
	assert (os.path.exists(os.path.join(args.data_dir, args.batch_dir)))

	agent = GetRewards(args)
	agent.run()

if __name__ == '__main__':

	main()
