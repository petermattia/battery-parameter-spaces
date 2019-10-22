from scipy.special import logsumexp
from functools import partial
import numpy as np
import time


def fit_model(model, x_train, y_train):
    success = model.fit(x_train, y_train)
    return (success, model)

def model_log_likelihood(model, x_test, y_test):
    return model.posterior_log_likelihood(x_test, y_test)

def model_posterior_prob_x_greater_than(model, x, y):
    return model.posterior_prob_x_greater_than(x, y)

def train_test_split(x, y, train_fraction):
    #split into train/test
    if train_fraction > 0.99:
        x_train = x
        y_train = y
        x_test = x
        y_test = y
    else:
        num_train = int(train_fraction * len(x))
        x_train = x[:num_train]
        y_train = y[:num_train]
        x_test = x[num_train:]
        y_test = y[num_train:]

    return x_train, y_train, x_test, y_test


class Ensemble(object):
    """
    """

    def __init__(self, models, map=map):
        """
            models: ensemble models
            map: map function, if multiprocessing is desired
        """
        self.all_models = models
        self._map = map
        self.fit_models = []


class CurveModelEnsemble(Ensemble):
    """

    """

    def __init__(self, models, map=map):
        super(CurveModelEnsemble, self).__init__(models, map)

    def fit(self, x, y, train_fraction=0.8):
        assert len(x) == len(y)

        model_log_likelihoods = []
        self.fit_models = []

        x_train, y_train, x_test, y_test = train_test_split(x, y, train_fraction)

        fit_result = self._map(
            partial(fit_model, x_train=x_train, y_train=y_train),
            self.all_models)
        for success, model in fit_result:
            if success:
                self.fit_models.append(model)

        if len(self.fit_models) == 0:
            logging.warn("EnsembleCurveModel couldn't fit any models!!!")
            return False

        model_log_likelihoods = self._map(
            partial(model_log_likelihood, x_test=x_test, y_test=y_test),
            self.fit_models)

        normalizing_constant = logsumexp(model_log_likelihoods)

        self.model_probabilities = [np.exp(log_lik - normalizing_constant) for log_lik in model_log_likelihoods]
        return True

    def posterior_prob_x_greater_than(self, x, y):
        """
            The probability under the models that a value y is exceeded at position x.
            IMPORTANT: if all models fail, by definition the probability is 1.0 and NOT 0.0
        """
        if len(self.fit_models) == 0:
            return 1.0

        models_prob_x_greater_than = model_log_likelihoods = self._map(
            partial(model_posterior_prob_x_greater_than, x=x, y=y),
            self.fit_models)

        overall_prob = 0
        for prob_x_greater_than, model_prob in zip(models_prob_x_greater_than, self.model_probabilities):
            overall_prob += model_prob * prob_x_greater_than
        return overall_prob

    def posterior_log_likelihood(self, x, y):
        log_liks = []
        for model, model_prob in zip(self.fit_models, self.model_probabilities):
            log_lik = model.posterior_log_likelihood(x, y)
            log_liks.append(np.log(model_prob) + log_lik)
        return logsumexp(log_liks)

    def predict(self, x):
        if np.isscalar(x):
            y = 0
        else:
            y = np.zeros(x.shape)
        for model, model_prob in zip(self.fit_models, self.model_probabilities):
            y += model_prob * model.predict(x)
        return y

    def __str__(self):
        ret = []
        model_names = [model.function.__name__ for model in self.fit_models]
        for model_prob, model_name in zip(self.model_probabilities, model_names):
            ret.append("%s: %f\n" % (model_name, model_prob))
        return "".join(ret)

    def serialize(self, fname):
        import pickle
        map_tmp = self._map
        self._map = None
        pickle.dump(self, open(fname, "wb"))
        self._map = map_tmp



class CurveEnsemble(Ensemble):
    """
        1. MCMC fitting
        2. Now take each theta as a model
        3. Make predictions as an weighted average of those models
            The weight is the (normalized) likelihood of some held out validation data.
    """

    def __init__(self, models, map=map):
        super(CurveEnsemble, self).__init__(models, map)

    def fit(self, x, y, train_fraction=0.8):
        assert len(x) == len(y)

        model_log_likelihoods = []
        self.fit_models = []

        x_train, y_train, x_test, y_test = train_test_split(x, y, train_fraction)

        fit_result = self._map(
            partial(fit_model, x_train=x_train, y_train=y_train),
            self.all_models)
        for success, model in fit_result:
            if success:
                self.fit_models.append(model)

        if len(self.fit_models) == 0:
            logging.warn("EnsembleCurveModel couldn't fit any models!!!")
            return

        #Now we interpret each theta as a separate model
        #TODO: parallelize!
        all_log_likelihoods = []
        for model in self.fit_models:
            model_log_likelihoods = []
            thetas = model.get_burned_in_samples()
            for theta_idx in range(thetas.shape[0]):
                theta = thetas[theta_idx,:]
                log_likelihood = model.ln_likelihood(theta, x_test, y_test)
                model_log_likelihoods.append(log_likelihood)
            all_log_likelihoods.append(model_log_likelihoods)

        self.model_theta_probabilities = [np.exp(model_log_likelihoods - logsumexp(model_log_likelihoods)) for model_log_likelihoods in all_log_likelihoods]

        normalizing_constant = logsumexp(all_log_likelihoods)

        normalize = lambda log_lik: np.exp(log_lik - normalizing_constant)

        self.all_model_probabilities = [[normalize(log_lik) for log_lik in model_log_likelihoods] for model_log_likelihoods in all_log_likelihoods]

        #sum up on a per model family basis:
        self.model_probabilities = [sum(model_probabilities) for model_probabilities in self.all_model_probabilities]


    def posterior_prob_x_greater_than(self, x, y):
        """
            The probability under the models that a value y is exceeded at position x.
            IMPORTANT: if all models fail, by definition the probability is 1.0 and NOT 0.0
        """
        if len(self.fit_models) == 0:
            return 1.0

        overall_prob = 0
        for model, theta_model_probabilities in zip(self.fit_models, self.all_model_probabilities):
            thetas = model.get_burned_in_samples()
            for theta_idx, theta_model_probability in zip(list(range(thetas.shape[0])), theta_model_probabilities):
                theta = thetas[theta_idx, :]
                overall_prob += theta_model_probability * model.prob_x_greater_than(x, y, theta)
        return overall_prob


    def predict(self, x):
        if np.isscalar(x):
            y = 0
        else:
            y = np.zeros(x.shape)
        #TOOD: implement!!
        return y

    def __str__(self):
        ret = []
        model_names = [model.function.__name__ for model in self.fit_models]
        for model_prob, model_name in zip(self.model_probabilities, model_names):
            ret.append("%s: %f\n" % (model_name, model_prob))
        return "".join(ret)