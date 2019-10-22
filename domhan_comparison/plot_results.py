import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pandas as pd

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
	# plt.plot(curve_data.T, color='tab:blue')
	# plt.show()

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


# def get_all_x_lifetimes(feature_filename=None,
#  lifetimes_filename=None, num_timesteps=10000):

# 	if feature_filename is None:
# 		feature_filename = os.path.join(filepath, 'IvsSOC_train.csv')
# 	feature_data = pd.read_csv(feature_filename, sep=',', header=None).to_numpy()
# 	f = np.repeat(feature_data, [num_timesteps]*feature_data.shape[0], axis=0)
# 	t = (np.repeat(1+np.arange(num_timesteps), feature_data.shape[0])).astype(np.float)
# 	x = np.concatenate((f, t[:, None]), axis=1)
# 	lifetimes = np.genfromtxt(lifetimes_filename, delimiter=',')
# 	return x, lifetimes

num_timesteps = 2300
# x_valid_all, lifetimes = get_all_x_lifetimes(feature_filename='IvsSOC_validation.csv',
# 	 lifetimes_filename='y_validation.csv',
# 	 num_timesteps=num_timesteps)


x_train, y_train = get_x_y(curve_filename='Qn_train.csv',
	feature_filename='IvsSOC_train.csv')


train_pred = np.load(sys.argv[1])
train_pred_mean = train_pred['mean']


valid_pred = np.load(sys.argv[2])
valid_pred_mean = (1.1-valid_pred['mean']).reshape((2300, -1)).T
print(valid_pred_mean.shape)
# print(train_data['mean'].shape, train_data['variance'].shape,
#  valid_data['mean'].shape, valid_data['variance'].shape)


train_file = glob.glob('Qn_train.csv')[0]
train = np.genfromtxt(train_file,delimiter=',')

valid_file = glob.glob('Qn_valid.csv')[0]
valid = np.genfromtxt(valid_file,delimiter=',')
print(valid.shape)

plt.plot(valid.T, color='tab:blue')
plt.plot(valid_pred_mean.T, color='tab:red')
# plt.plot(valid.T,color='tab:red')

plt.legend(frameon=False)
plt.savefig('nn_fits.png', bbox_inches='tight')
plt.show()