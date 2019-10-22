import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch

np.random.seed(0)
torch.manual_seed(0)

lifetime_scaling = 2500.
q_scaling = 100
max_q = 1.1
time_limit = 100

def get_x_y(curve_filename, feature_filename, filepath='./'):

	if curve_filename != 'Qn_test2.csv':
		curve_data = pd.read_csv(curve_filename, sep=',', header=None).to_numpy()
	else:
		curve_data = pd.read_csv(curve_filename, sep='\t', header=None).to_numpy()

	curve_data[np.where(curve_data<0.88)] = np.NaN

	q = q_scaling * (max_q - curve_data.flatten())
	t = (np.tile(1+np.arange(curve_data.shape[1]), curve_data.shape[0])
		.astype(np.float))/lifetime_scaling
	print(curve_data.shape, t.shape, q.shape)

	feature_data = pd.read_csv(feature_filename, sep=',', header=None).to_numpy()
	print(feature_data.shape)
	f = np.repeat(feature_data, [curve_data.shape[1]]*feature_data.shape[0], axis=0)

	nonnan_idx = np.argwhere(np.logical_not(np.isnan(q)))[:, 0]
	x = np.concatenate((f, q[:, None]), axis=1)[nonnan_idx]
	# x = np.concatenate((f, y[:, None]), axis=1)[nonnan_idx]
	t = t[nonnan_idx]
	print('max t, min t', np.amax(t), np.amin(t), np.argmax(t), np.argmin(t))
	print(nonnan_idx.shape, curve_data.shape, feature_data.shape, f.shape, q.shape, x.shape, t.shape)
	return x, t


def get_all_x_lifetimes(feature_filename, lifetimes_filename):

	feature_data = pd.read_csv(feature_filename, sep=',', header=None).to_numpy()
	q = (np.repeat([q_scaling * (max_q - 0.88)], feature_data.shape[0])).astype(np.float)
	x = np.concatenate((feature_data, q[:, None]), axis=1)
	lifetimes = pd.read_csv(lifetimes_filename, sep=',', header=None).to_numpy()
	print(x.shape, lifetimes.shape)
	return x, lifetimes[:, 0]


def main():

	# Load data.
	x_train, y_train = get_x_y(curve_filename='Qn_train.csv',
	 feature_filename='IvsSOC_train.csv')
	x_test, y_test = get_x_y(curve_filename='Qn_test1.csv',
	 feature_filename='IvsSOC_test1.csv')
	# Initialize and learn model.
	# import pdb
	# pdb.set_trace()

	from pybnn.lcnet import LCNet
	# model = LCNet(normalize_input=True, normalize_output=False)
	model = LCNet(normalize_input=False, normalize_output=False)

	# torch.save(model, 'best_model.pt')
	# model = torch.load('best_model.pt')

	# model.train(x, y, verbose=True)
	# model.train_and_evaluate(x_valid, y_valid, x_valid, y_valid,
	#  verbose=True, lr=0.0005, batch_size=2000, num_burn_in_steps=10000)
	model.train_and_evaluate(x_train, y_train, x_test, y_test,
	 verbose=True, lr=0.005,
	  batch_size=1000,
	  num_steps=13000,
	  num_burn_in_steps=3000,
	  validate_every_n_steps=1000)
	# 500,300,200
	# 13k, 3k, 1k

	x_train_all, train_lifetimes = get_all_x_lifetimes(feature_filename='IvsSOC_train.csv',
	 lifetimes_filename='y_train.csv')
	mt, vt = model.predict(x_train_all)
	print(lifetime_scaling*mt, train_lifetimes)
	print('Train RMSE', np.sqrt(mean_squared_error(lifetime_scaling*mt, train_lifetimes)))

	x_test_all, test_lifetimes = get_all_x_lifetimes(feature_filename='IvsSOC_test1.csv',
	 lifetimes_filename='y_test1.csv')
	mte, vte = model.predict(x_test_all)
	print(lifetime_scaling*mte, test_lifetimes)
	print('Test RMSE', np.sqrt(mean_squared_error(lifetime_scaling*mte, test_lifetimes)))

	x_test_all, test2_lifetimes = get_all_x_lifetimes(feature_filename='IvsSOC_test2.csv',
	 lifetimes_filename='y_test2.csv')
	mte2, vte2 = model.predict(x_test_all)
	print(lifetime_scaling*mte2, test2_lifetimes)
	print('Test 2 RMSE', np.sqrt(mean_squared_error(lifetime_scaling*mte2, test2_lifetimes)))

	x_valid_all, valid_lifetimes = get_all_x_lifetimes(feature_filename='IvsSOC_validation.csv',
	 lifetimes_filename='y_validation.csv')
	mv, vv = model.predict(x_valid_all)
	print(lifetime_scaling*mv, valid_lifetimes)
	print('Validation RMSE', np.sqrt(mean_squared_error(lifetime_scaling*mv, valid_lifetimes)))


	np.savez('predictions_train', mean=mt, variance=vt)
	np.savez('predictions_valid', mean=mv, variance=vv)
	np.savez('predictions_test', mean=mte, variance=vte)
	np.savez('predictions_test2', mean=mte2, variance=vte2)



if __name__ == "__main__":
	main()