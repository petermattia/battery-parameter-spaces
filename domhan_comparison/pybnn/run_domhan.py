import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch

lifetime_scaling = 2500.
q_scaling = 100
max_q = 1.1
time_limit = 100

def get_x_y(curve_data, feature_data):

	t = (1+np.arange(time_limit))/lifetime_scaling
	q = q_scaling * (max_q-curve_data[:time_limit])
	f = np.repeat(feature_data[None, :], [len(q)], axis=0)
	nonnan_idx = np.argwhere(np.logical_not(np.isnan(q)))[:, 0]
	x = np.concatenate((f, q[:, None]), axis=1)[nonnan_idx]
	t = t[nonnan_idx]
	print('max t, min t', np.amax(t), np.amin(t), np.argmax(t), np.argmin(t))
	print(nonnan_idx.shape, curve_data.shape, feature_data.shape, f.shape, q.shape, x.shape, t.shape)

	final_q = np.array([q_scaling * (max_q-0.88)])
	x_test = np.concatenate((feature_data[None, :], final_q[:, None]), axis=1)

	return x, t, x_test

def main():

	from pybnn.lcnet import LCNet

	curve_filenames = ['Qn_train.csv', 'Qn_test1.csv', 'Qn_test2.csv']
	feature_filenames = ['IvsSOC_train.csv', 'IvsSOC_test1.csv', 'IvsSOC_test2.csv']
	lifetime_filenames = ['y_train.csv', 'y_test1.csv', 'y_test2.csv']

	for curve_filename, feature_filename, lifetime_filename in zip(curve_filenames, feature_filenames, lifetime_filenames):
		print(curve_filename, feature_filename, lifetime_filename)
		# Load data.
		sep = ','
		if curve_filename == 'Qn_test2.csv':
			sep = '\t'
		if curve_filename == 'Qn_train.csv':
			continue

		curve_data = pd.read_csv(curve_filename, sep=sep, header=None).to_numpy()
		feature_data = pd.read_csv(feature_filename, sep=',', header=None).to_numpy()
		lifetimes = pd.read_csv(lifetime_filename, sep=',', header=None).to_numpy()

		num_curves = curve_data.shape[0]
		all_rmse = []
		for i in range(num_curves):
			print('Curve', i)
			x_train, y_train, x_test = get_x_y(curve_data[i], feature_data[i])
			y_test = [lifetimes[i]/lifetime_scaling]
			# import pdb; pdb.set_trace()
			model = LCNet(normalize_input=False, normalize_output=False)
			_, _, mses = model.train_and_evaluate(x_train, y_train, x_test, y_test,
			  verbose=True, lr=0.005,
			  batch_size=100,
			  num_steps=1000,
			  num_burn_in_steps=3000,
			  validate_every_n_steps=200)
			all_rmse.append(lifetime_scaling * np.sqrt(mses[-1]))
		# 500,300,200
		# 13k, 3k, 1k

		print('all rmses', all_rmse)
		print('avg rmse', np.mean(all_rmse))

if __name__ == "__main__":
	main()