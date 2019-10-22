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

from caffe.proto import caffe_pb2
import google
from google.protobuf import text_format

import numpy as np

from modelfactory import create_model, setup_model_combination

IMPROVEMENT_PROB_THRESHOLD = 0.05
PREDICTIVE_STD_THRESHOLD = 0.005

PREDICTION_THINNING = 10
NTHREADS = 4


def cut_beginning(y, threshold=0.05, look_ahead=5):
    """
        we start at a point where we are bigger than the initial value for look_ahead steps
    """
    if len(y) < look_ahead:
        return y
    num_cut = 0
    for idx in range(len(y)-look_ahead):
        start_here = True
        for idx_ahead in range(idx, idx+look_ahead):
            if not (y[idx_ahead] - y[0] > threshold):
                start_here = False
        if start_here:
            num_cut = idx
            break
    return y[num_cut:]


def get_xlim():
    assert os.path.exists("caffenet_solver.prototxt")
    solver = caffe_pb2.SolverParameter()
    solver_txt = open("caffenet_solver.prototxt").read()
    try:
        google.protobuf.text_format.Merge(solver_txt, solver)
    except Exception as e:
        #this may happen if fields are added. However everything else should be parse
        #hence, it's ok to be ignored
        print(("error parsing solver: ", str(e)))
    assert solver.max_iter > 0
    assert solver.test_interval > 0
    return solver.max_iter / float(solver.test_interval)


class TerminationCriterion(object):
    def __init__(self, nthreads, prob_x_greater_type):
        open("helloworld", "w").write("test")
        self.prob_x_greater_type = prob_x_greater_type
        print(("prob_x_greater_type: %s" % prob_x_greater_type))
        #just make sure there is no y_predict file from previous runs:
        if os.path.exists("y_predict.txt"):
            os.remove("y_predict.txt")
        models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
                  "mmf", "janoschek", "dr_hill_zero_background", "log_power",
                  "exp4"]
        xlim = get_xlim()
        print(("xlim: %d" % (xlim)))
        self.xlim = xlim
        model = setup_model_combination(#create_model(
            #"curve_combination",
            models=models,
            xlim=xlim,
            recency_weighting=True,
            nthreads=nthreads)
        self.model = model
    
    def run(self):
        pass

    def predict(self):
        """
            predict f(x), returns 1 if not successful
        """
        #we're are most likely not going to improve, stop!
        #let's made a prediction of the accuracy that will most likely be reached, that will be returned to the optimizer
        y_predict = self.model.predict(self.xlim, thin=PREDICTION_THINNING)
        #let's do a sanity check:
        if y_predict >= 0. and y_predict <= 1.0:
            with open("y_predict.txt", "w") as y_predict_file:
                y_predict_file.write(str(y_predict))
            print(("probably only going to reach %f, stopping..." % y_predict))
            return 1
        else:
            #we did not pass the sanity check.. let's not report this to the optimizer
            #and pretend nothing happened
            print(("didn't pass sanity check with predicted value %f" % y_predict))
            return 0


class ConservativeTerminationCriterion(TerminationCriterion):
    """
        Will evaluate p(y > y_best) and stop if the result doesn't look promising.
        In any other case we will continue running.
    """
    def __init__(self, nthreads, prob_x_greater_type, predictive_std_threshold=None):
        super(ConservativeTerminationCriterion, self).__init__(nthreads, prob_x_greater_type)
        self.predictive_std_threshold = predictive_std_threshold

    def run(self):
        if not os.path.exists("ybest.txt"):
            #no ybest yet... we can't do much
            print("not ybest yet...exiting")
            return 0
        ybest = float(open("ybest.txt").read())
        assert os.path.exists("learning_curve.txt"), "no learning_curve.txt ... nothing to do"

        y = np.loadtxt("learning_curve.txt")

        y_curr_best = np.max(y)

        if y_curr_best > ybest:
            #we already exceeded ybest ... let the other criterions decide when to stop
            print("Already better than ybest... not evaluating f(y)>f(y_best)")
            return 0

        #TODO subtract num_cut from xlim!
        y = cut_beginning(y)
        x = np.asarray(list(range(1, len(y)+1)))

        if not self.model.fit(x, y):
            #failed fitting... not cancelling
            print("failed fitting the model")
            return 0

        if self.prob_x_greater_type == "posterior_prob_x_greater_than":
            prob_gt_ybest_xlast = self.model.posterior_prob_x_greater_than(self.xlim,
                ybest, thin=PREDICTION_THINNING)
        else:
            prob_gt_ybest_xlast = self.model.posterior_mean_prob_x_greater_than(self.xlim,
                ybest, thin=PREDICTION_THINNING)

        print(("p(y>y_best) = %f" % prob_gt_ybest_xlast))

        if prob_gt_ybest_xlast < IMPROVEMENT_PROB_THRESHOLD:
            if self.predictive_std_threshold is None:
                return self.predict()
            else:
                print("predictive_std_threshold set, checking the predictive_std first")
                predictive_std = self.model.predictive_std(self.xlim, thin=PREDICTION_THINNING)
                print(("predictive_std: %f" % predictive_std))

                if predictive_std < self.predictive_std_threshold:
                    print("predicting...")
                    return self.predict()
                else:
                    print("continue evaluating")
                    #we are gonna wait before we become more certain about the outcome!
                    return 0
            
        else:
            print("continue evaluating")
            #we are probably still going to improve
            return 0


class OptimisticTerminationCriterion(TerminationCriterion):
    """
        Similar to the ConservativeTerminationCriterion will evaluate p(y > y_best)
        and stop if the result doesn't look promising.
        However additionally, if the model is confident in the prediction we will still
        stop to save time, at the risk of making a wrong prediction that we take.
    """
    def __init__(self, nthreads,
            prob_x_greater_type,
            predictive_std_threshold=PREDICTIVE_STD_THRESHOLD):
        if predictive_std_threshold is None:
            predictive_std_threshold = PREDICTIVE_STD_THRESHOLD
        assert predictive_std_threshold > 0
        self.predictive_std_threshold = predictive_std_threshold
        super(OptimisticTerminationCriterion, self).__init__(nthreads, prob_x_greater_type)


    def run(self):
        if not os.path.exists("learning_curve.txt"):
            print("no learning_curve.txt ... nothing to do")
            return 0

        if os.path.exists("ybest.txt"):
            ybest = float(open("ybest.txt").read())
        else:
            ybest = None
        y = np.loadtxt("learning_curve.txt")

        y_curr_best = np.max(y)

        #TODO subtract num_cut from xlim!
        y = cut_beginning(y)
        x = np.asarray(list(range(1, len(y)+1)))

        if not self.model.fit(x, y):
            #failed fitting... not cancelling
            return 0

        predictive_std = self.model.predictive_std(self.xlim, thin=PREDICTION_THINNING)
        print(("predictive_std: %f" % predictive_std))

        if predictive_std < self.predictive_std_threshold:
            #the model is pretty sure about the prediction: stop!
            print("predictive_std low, predicting and stopping...")
            return self.predict()
        elif ybest is not None:
            print("predictive_std high, let's check the probably to get higher than the current ybest")
            #we're still checking if maybe all the probability is below ybest
            if self.prob_x_greater_type == "posterior_prob_x_greater_than":
                prob_gt_ybest_xlast = self.model.posterior_prob_x_greater_than(self.xlim,
                    ybest, thin=PREDICTION_THINNING)
            else:
                prob_gt_ybest_xlast = self.model.posterior_mean_prob_x_greater_than(self.xlim,
                    ybest, thin=PREDICTION_THINNING)

            print(("p(y>y_best) = %f" % prob_gt_ybest_xlast))

            if prob_gt_ybest_xlast < IMPROVEMENT_PROB_THRESHOLD:
                return self.predict()
            else:
                print("continue evaluating")
                #we are probably still going to improve
                return 0
        else:
            print("neither the predictive_std is low nor is there a ybest ... continue")
            return 0




def main(mode="conservative",
    prob_x_greater_type="posterior_prob_x_greater_than",
    nthreads=NTHREADS,
    predictive_std_threshold=None):
    ret = 0
    try:
        open("termination_criterion_running_pid", "w").write(str(os.getpid()))

        assert prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"], ("prob_x_greater_type unkown %s" % prob_x_greater_type)

        #ret = run_prediction(nthreads)
        #return ret
        if mode == "conservative":
            term_crit = ConservativeTerminationCriterion(nthreads,
                prob_x_greater_type,
                predictive_std_threshold=predictive_std_threshold)
            ret = term_crit.run()
        elif mode == "optimistic":
            term_crit = OptimisticTerminationCriterion(nthreads,
                prob_x_greater_type,
                predictive_std_threshold=predictive_std_threshold)
            ret = term_crit.run()
        else:
            print("The mode can either be conservative or optimistic")
            ret = 0
    except Exception as e:
        import traceback
        with open("term_crit_error.txt", "a") as error_log:
            error_log.write(str(traceback.format_exc()))
            error_log.write(str(e))
    finally:
        if os.path.exists("termination_criterion_running"):
            os.remove("termination_criterion_running")
        if os.path.exists("termination_criterion_running_pid"):
            os.remove("termination_criterion_running_pid")
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Termination criterion.')
    parser.add_argument('--nthreads', type=int, default=1, help='number of threads to launch')
    parser.add_argument('--mode', type=str, default="conservative", help='either conservative or optimistic')
    parser.add_argument('--prob-x-greater-type', type=str, default="posterior_prob_x_greater_than", help='either posterior_mean_prob_x_greater_than or posterior_prob_x_greater_than')
    parser.add_argument('--predictive-std-threshold', type=float,
        default=None, help='threshold for making optimistic guesses about the learning curve.')

    args = parser.parse_args()

    ret = main(mode=args.mode, prob_x_greater_type=args.prob_x_greater_type,
        nthreads=args.nthreads, predictive_std_threshold=args.predictive_std_threshold)

    print(("exiting with status: %d" % ret))
    sys.exit(ret)
