"""
    Input:
        learning_curve.txt a file that contains the current learning curve
        ybest.txt the best result achieved so far
        xlim.txt the x position we are targetting

    Output:
        y_predict.txt prediction of performance at xlim (in case the exit code is 1)


    Reads learning_curve.txt and make a prediction whether to cancel the current run,
    based on the likelihood that we will exceed ybest at xlim. The result will
    be set in the return code.

    Returns:
        exit code 0 means: continue running the algorithm.
        exit code 1 means: save to cancel the run.
"""
#TODO: remove num_cut from xlim!
import os
import sys
import argparse
import numpy as np

from modelfactory import create_model, setup_model_combination

NTHREADS = 4

def get_x_y(filename=None):
    if filename is None:
        filename = 'Qn_train.csv'
    data = np.genfromtxt(filename, delimiter=',')

    y = data.flatten()
    x = (np.repeat(1+np.arange(data.shape[1]), data.shape[0])
        .astype(np.float))
    print(data.shape, x.shape, y.shape)
    
    nonnan_idx = np.argwhere(np.logical_not(np.isnan(y)))
    x = x[nonnan_idx][:, 0]
    y = y[nonnan_idx][:, 0]
    print(data.shape, x.shape, y.shape)
    return x, y


def main(nthreads=NTHREADS):

    # Warm start model via MLE fitting

    models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
              "mmf", "janoschek", "dr_hill_zero_background", "log_power",
              "exp4"]
    xlim = 2500
    model = setup_model_combination(models=models,
            xlim=xlim,
            recency_weighting=True,
            nthreads=nthreads)

    x, y = get_x_y()

    if not model.fit(x, y):
        #failed fitting... not cancelling
        print("failed fitting the model")
        return 0
        

    # Setup Regressor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Termination criterion.')
    parser.add_argument('--nthreads', type=int, default=1, help='number of threads to launch')
    args = parser.parse_args()

    main(nthreads=args.nthreads)
