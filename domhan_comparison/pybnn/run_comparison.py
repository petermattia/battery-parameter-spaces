import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_x_y(filepath='./',
		curve_filename=None,
		feature_filename=None):
	if curve_filename is None:
		curve_filename = os.path.join(filepath, 'Qn_train.csv')
	curve_data = pd.read_csv(curve_filename, sep=',', header=None).to_numpy()
	# curve_data[np.where(curve_data<0.88)] = np.NaN
	# curve_data = curve_data[2][None, :]
	# np.genfromtxt(curve_filename, delimiter=',')

	y = 1.1-curve_data.flatten()
	t = (np.repeat(1+np.arange(curve_data.shape[1]), curve_data.shape[0])
		.astype(np.float))
	print(curve_data.shape, t.shape, y.shape)
	plt.plot(curve_data.T, color='tab:blue')
	plt.show()

	if feature_filename is None:
		feature_filename = os.path.join(filepath, 'IvsSOC_train.csv')
	feature_data = pd.read_csv(feature_filename, sep=',', header=None).to_numpy()
	print(feature_data.shape)
	f = np.repeat(feature_data, [curve_data.shape[1]]*feature_data.shape[0], axis=0)

	nonnan_idx = np.argwhere(np.logical_not(np.isnan(y)))[:, 0]
	x = np.concatenate((f, t[:, None]), axis=1)[nonnan_idx]
	# x = np.concatenate((f, y[:, None]), axis=1)[nonnan_idx]
	y = y[nonnan_idx]
	print('max y, min y', np.amax(y), np.amin(y), np.argmax(y), np.argmin(y))
	print(nonnan_idx.shape, curve_data.shape, feature_data.shape, f.shape, t.shape, x.shape, y.shape)
	return x, y


def get_all_x_lifetimes(feature_filename=None,
 lifetimes_filename=None, num_timesteps=10000):

	if feature_filename is None:
		feature_filename = os.path.join(filepath, 'IvsSOC_train.csv')
	feature_data = pd.read_csv(feature_filename, sep=',', header=None).to_numpy()
	f = np.repeat(feature_data, [num_timesteps]*feature_data.shape[0], axis=0)
	t = (np.repeat(1+np.arange(num_timesteps), feature_data.shape[0])).astype(np.float)
	x = np.concatenate((f, t[:, None]), axis=1)
	lifetimes = np.genfromtxt(lifetimes_filename, delimiter=',')
	return x, lifetimes


def main():

	# Load data.
	x_train, y_train = get_x_y(curve_filename='Qn_train.csv',
	 feature_filename='IvsSOC_train.csv')
	x_valid, y_valid = get_x_y(curve_filename='Qn_valid.csv',
	 feature_filename='IvsSOC_validation.csv')
	# Initialize and learn model.

	from pybnn.lcnet import LCNet
	# model = LCNet(normalize_input=True, normalize_output=False)
	model = LCNet(normalize_input=False, normalize_output=False)

	# model.train(x, y, verbose=True)
	# model.train_and_evaluate(x_valid, y_valid, x_valid, y_valid,
	#  verbose=True, lr=0.0005, batch_size=2000, num_burn_in_steps=10000)
	model.train_and_evaluate(x_train, y_train, x_valid, y_valid,
	 verbose=True, lr=0.0005,
	  batch_size=500,
	  num_steps=5000,
	  num_burn_in_steps=3000,
	  validate_every_n_steps=1000)
	# import pdb
	# pdb.set_trace()
	mt, vt = model.predict(x_train)

	num_timesteps = 2300
	x_valid_all, lifetimes = get_all_x_lifetimes(feature_filename='IvsSOC_validation.csv',
	 lifetimes_filename='y_validation.csv',
	 num_timesteps=num_timesteps)
	
	mv, vv = model.predict(x_valid_all)

	# import pdb
	# pdb.set_trace()

	np.savez('predictions_train_100', mean=mt, variance=vt)
	np.savez('predictions_valid_100', mean=mv, variance=vv)


	





if __name__ == "__main__":
	main()