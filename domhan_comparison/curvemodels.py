import numpy as np
import emcee
import inspect
import traceback
from scipy.stats import norm, kde
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
import lmfit
import logging
from scipy.special import logsumexp
from curvefunctions import all_models
from functools import reduce

def recency_weights(num):
    if num == 1:
        return np.ones(1)
    else:
        recency_weights = [10**(1./num)] * num
        recency_weights = recency_weights**(np.arange(0, num))
        return recency_weights

def masked_mean_x_greater_than(posterior_distribution, y):
    """
        P(E[f(x)] > E[y] | Data)
    """
    predictions = np.ma.masked_invalid(posterior_distribution)
    return np.sum(predictions > y) / float(np.sum(~predictions.mask))


class CurveModel(object):

    def __init__(self,
                 function,
                 function_der=None,
                 min_vals={},
                 max_vals={},
                 default_vals={}):
        """
            function: the function to be fit
            function_der: derivative of that function
        """
        self.function = function
        if function_der != None:
            raise NotImplementedError("function derivate is not implemented yet...sorry!")
        self.function_der = function_der
        assert isinstance(min_vals, dict)
        self.min_vals = min_vals.copy()
        assert isinstance(max_vals, dict)
        self.max_vals = max_vals.copy()
        function_args = inspect.getargspec(function).args
        assert "x" in function_args, "The function needs 'x' as a parameter."
        for default_param_name in list(default_vals.keys()):
            if default_param_name == "sigma":
                continue
            msg = "function %s doesn't take default param %s" % (function.__name__, default_param_name)
            assert default_param_name in function_args, msg
        self.function_params = [param for param in function_args if param != 'x']
        #set default values:
        self.default_vals = default_vals.copy()
        for param_name in self.function_params:
            if param_name not in default_vals:
                print("setting function parameter %s to default of 1.0 for function %s" % (param_name,
                                                                                           self.function.__name__))
                self.default_vals[param_name] = 1.0
        self.all_param_names = [param for param in self.function_params]
        self.all_param_names.append("sigma")
        self.name = self.function.__name__
        self.ndim = len(self.all_param_names)
        
        #uniform noise prior over interval:
        if "sigma" not in self.min_vals:
            self.min_vals["sigma"] = 0.
        if "sigma" not in self.max_vals:
            self.max_vals["sigma"] = 1.0
        if "sigma" not in self.default_vals:
            self.default_vals["sigma"] = 0.05
    
    def default_function_param_array(self):
        return np.asarray([self.default_vals[param_name] for param_name in self.function_params])

    def are_params_in_bounds(self, theta):
        """
            Are the parameters in their respective bounds?
        """
        in_bounds = True
        
        for param_name, param_value in zip(self.all_param_names, theta):
            if param_name in self.min_vals:
                if param_value < self.min_vals[param_name]:
                    in_bounds = False
            if param_name in self.max_vals:
                if param_value > self.max_vals[param_name]:
                    in_bounds = False
        return in_bounds

    def split_theta(self, theta):
        """Split theta into the function parameters (dict) and sigma. """
        params = {}
        sigma = None
        for param_name, param_value in zip(self.all_param_names, theta):
            if param_name in self.function_params:
                params[param_name] = param_value
            elif param_name == "sigma":
                sigma = param_value
        return params, sigma

    def split_theta_to_array(self, theta):
        """Split theta into the function parameters (array) and sigma. """
        params = theta[:-1]
        sigma = theta[-1]
        return params, sigma

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def predict_given_theta(self, x, theta):
        """
            Make predictions given a single theta
        """
        params, sigma = self.split_theta(theta)
        predictive_mu = self.function(x, **params)
        return predictive_mu, sigma

    def likelihood(self, x, y):
        """
            for each y_i in y:
                p(y_i|x, model)
        """
        params, sigma = self.split_theta(self.ml_params)
        return norm.pdf(y-self.function(x, **params), loc=0, scale=sigma)


class MLCurveModel(CurveModel):
    """
        ML fit of a curve.
    """

    def __init__(self, recency_weighting=True,  **kwargs):
        super(MLCurveModel, self).__init__(**kwargs)

        #Maximum Likelihood values of the parameters
        self.ml_params = None
        self.recency_weighting = recency_weighting

    def fit(self, x, y, weights=None, start_from_default=True):
        """
            weights: None or weight for each sample.
        """
        return self.fit_ml(x, y, weights, start_from_default)

    def predict(self, x):
        #assert len(x.shape) == 1
        params, sigma = self.split_theta_to_array(self.ml_params)
        return self.function(x, *params)
        #return np.asarray([self.function(x_pred, **params) for x_pred in x])

    def fit_ml(self, x, y, weights, start_from_default):
        """
            non-linear least-squares fit of the data.

            First tries Levenberg-Marquardt and falls back
            to BFGS in case that fails.

            Start from default values or from previous ml_params?
        """
        # successful = self.fit_leastsq(x, y, weights, start_from_default)
        successful = False
        if not successful:
            successful = self.fit_bfgs(x, y, weights, start_from_default)
            if not successful:
                return False
        return successful

    def ml_sigma(self, x, y, popt, weights):
        """
            Given the ML parameters (popt) get the ML estimate of sigma.
        """
        if weights is None:
            if self.recency_weighting:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=recency_weights(len(y)))
                sigma = np.sqrt(variance)
            else:
                sigma = (y-self.function(x, *popt)).std()
        else:
            if self.recency_weighting:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=recency_weights(len(y)) * weights)
                sigma = np.sqrt(variance)
            else:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=weights)
                sigma = np.sqrt(variance)
        return sigma

    def fit_leastsq(self, x, y, weights, start_from_default):    
        try:
            if weights is None:
                if self.recency_weighting:
                    residuals = lambda p: np.sqrt(recency_weights(len(y))) * (self.function(x, *p) - y)
                else:
                    residuals = lambda p: self.function(x, *p) - y
            else:
                #the return value of this function will be squared, hence
                #we need to take the sqrt of the weights here
                if self.recency_weighting:
                    residuals = lambda p: np.sqrt(recency_weights(len(y))*weights) * (self.function(x, *p) - y)
                else:
                    residuals = lambda p: np.sqrt(weights) * (self.function(x, *p) - y)

            
            if start_from_default:
                initial_params = self.default_function_param_array()
            else:
                initial_params, _ = self.split_theta_to_array(self.ml_params)
            popt, cov_popt, info, msg, status = leastsq(residuals,
                    x0=initial_params,
                    full_output=True)
                #Dfun=,
                #col_deriv=True)
            
            if np.any(np.isnan(info["fjac"])):
                return False

            leastsq_success_statuses = [1,2,3,4]
            if status in leastsq_success_statuses:
                if any(np.isnan(popt)):
                    return False
                #within bounds?
                if not self.are_params_in_bounds(popt):
                    return False

                sigma = self.ml_sigma(x, y, popt, weights)
                self.ml_params = np.append(popt, [sigma])

                logging.info("leastsq successful for model %s" % self.function.__name__)

                return True
            else:
                logging.warn("leastsq NOT successful for model %s, msg: %s" % (self.function.__name__, msg))
                logging.warn("best parameters found: " + str(popt))
                return False
        except Exception as e:
            print(e)
            tb = traceback.format_exc()
            print(tb)
            return False

    def fit_bfgs(self, x, y, weights, start_from_default):
        try:
            def objective(params):
                if weights is None:
                    if self.recency_weighting:
                        return np.sum(recency_weights(len(y))*(self.function(x, *params) - y)**2)
                    else:
                        return np.sum((self.function(x, *params) - y)**2)
                else:
                    if self.recency_weighting:
                        return np.sum(weights * recency_weights(len(y)) * (self.function(x, *params) - y)**2)
                    else:
                        return np.sum(weights * (self.function(x, *params) - y)**2)
            bounds = []
            for param_name in self.function_params:
                if param_name in self.min_vals and param_name in self.max_vals:
                    bounds.append((self.min_vals[param_name], self.max_vals[param_name]))
                elif param_name in self.min_vals:
                    bounds.append((self.min_vals[param_name], None))
                elif param_name in self.max_vals:
                    bounds.append((None, self.max_vals[param_name]))
                else:
                    bounds.append((None, None))

            if start_from_default:
                initial_params = self.default_function_param_array()
            else:
                initial_params, _ = self.split_theta_to_array(self.ml_params)

            popt, fval, info= fmin_l_bfgs_b(objective,
                                            x0=initial_params,
                                            bounds=bounds,
                                            approx_grad=True)
            if info["warnflag"] != 0:
                logging.warn("BFGS not converged! (warnflag %d) for model %s" % (info["warnflag"], self.name))
                logging.warn(info)
                return False

            if popt is None:
                return False
            if any(np.isnan(popt)):
                logging.info("bfgs NOT successful for model %s, parameter NaN" % self.name)
                return False
            sigma = self.ml_sigma(x, y, popt, weights)
            self.ml_params = np.append(popt, [sigma])
            logging.info("bfgs successful for model %s" % self.name)
            return True
        except:
            return False

    def aic(self, x, y):
        """
            Akaike information criterion
            http://en.wikipedia.org/wiki/Akaike_information_criterion
        """
        params, sigma = self.split_theta_to_array(self.ml_params)
        y_model = self.function(x, *params)
        log_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()
        return 2 * len(self.function_params) - 2 * log_likelihood



class MCMCCurveModel(CurveModel):
    """
        MLE curve fitting + MCMC sampling with uniform priors for parameter uncertainty.

        Model: y ~ f(x) + eps with eps ~ N(0, sigma^2)
    """
    def __init__(self,
                 function,
                 function_der=None,
                 min_vals={},
                 max_vals={},
                 default_vals={},
                 burn_in=300,
                 nwalkers=100,
                 nsamples=800,
                 nthreads=1,
                 recency_weighting=False):
        """
            function: the function to be fit
            function_der: derivative of that function
        """
        super(MCMCCurveModel, self).__init__(
            function=function,
            function_der=function_der,
            min_vals=min_vals,
            max_vals=max_vals,
            default_vals=default_vals)
        self.ml_curve_model = MLCurveModel(
            function=function,
            function_der=function_der,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            default_vals=self.default_vals,
            recency_weighting=recency_weighting)

        #TODO: have two burn-ins, one for when the ML fitting is successful and one for when not!
        self.burn_in = burn_in
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.nthreads = nthreads
        self.recency_weighting = recency_weighting
    
    def fit(self, x, y):
        try:
            if self.ml_curve_model.fit(x, y):
                logging.info("ML fit: " + str(self.ml_curve_model.ml_params))
                self.fit_mcmc(x, y)
                return True
            else:
                return False
        except Exception as e:
            print(e)
            tb = traceback.format_exc()
            print(tb)
            return False

    #priors
    def ln_prior(self, theta):
        """
            log-prior is (up to a constant)
        """
        if self.are_params_in_bounds(theta):
            return 0.0
        else:
            return -np.inf
    
    #likelihood
    def ln_likelihood(self, theta, x, y):
        """
           y = y_true + y_noise
            with y_noise ~ N(0, sigma^2)
        """
        params, sigma = self.split_theta(theta)
        y_model = self.function(x, **params)
        if self.recency_weighting:
            weight = recency_weights(len(y))
            ln_likelihood = (weight*norm.logpdf(y-y_model, loc=0, scale=sigma)).sum()
        else:
            ln_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()
        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood
        
    def ln_prob(self, theta, x, y):
        """
            posterior probability
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y)
    
    def fit_mcmc(self, x, y):
        #initialize in an area around the starting position
        #pos = [start + 1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        assert self.ml_curve_model.ml_params is not None
        pos = [self.ml_curve_model.ml_params + 1e-6*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        if self.nthreads <= 1:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                self.ndim,
                self.ln_prob,
                args=(x, y))
        else:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                model_ln_prob,
                args=(self, x, y),
                threads=self.nthreads)
        sampler.run_mcmc(pos, self.nsamples)
        self.mcmc_chain = sampler.chain
        
    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            params, sigma = self.split_theta(theta)
            predictions.append(self.function(x, **params))
        return np.asarray(predictions)

    def predictive_ln_prob_distribution(self, x, y, thin=1):
        """
            posterior log p(y|x,D) for each sample
        """
        #assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        ln_probs = []
        for theta in samples[::thin]:
            ln_prob = self.ln_likelihood(theta, x, y)
            ln_probs.append(ln_prob)
        return np.asarray(ln_probs)

    def posterior_ln_prob(self, x, y, thin=10):
        """
            posterior log p(y|x,D)

            1/S sum p(y|D,theta_s)
            equivalent to:
            logsumexp(log p(y|D,theta_s)) - log(S)
        """
        assert not np.isscalar(x)
        assert not np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        ln_probs = self.predictive_ln_prob_distribution(x, y)
        #print ln_probs
        #print np.max(ln_probs)
        #print np.min(ln_probs)
        #print np.mean(ln_probs)
        #print "logsumexp(ln_probs)", logsumexp(ln_probs)
        #print "np.log(len(ln_probs)) ", np.log(len(ln_probs))
        #print logsumexp(ln_probs) - np.log(len(ln_probs))
        return logsumexp(ln_probs) - np.log(len(ln_probs))

    def predict(self, x):
        """
            E[f(x)]
        """
        predictions = self.predictive_distribution(x)
        return np.ma.masked_invalid(predictions).mean()
    
    def predictive_density(self, x_pos, x_density):
        density = kde.gaussian_kde(self.predictive_distribution(x_pos))
        return density(x_density)

    def prob_x_greater_than(self, x, y, theta):
        """
            P(f(x) > y | Data, theta)
        """
        params, sigma = self.split_theta(theta)
        mu = self.function(x, **params)
        cdf = norm.cdf(y, loc=mu, scale=sigma)
        return 1. - cdf

    def posterior_mean_prob_x_greater_than(self, x, y, thin=1):
        """
            P(E[f(x)] > E[y] | Data)

            thin: only use every thin'th sample
        
            Posterior probability that the expected valuef(x) is greater than 
            the expected value of y.
        """
        posterior_distribution = self.predictive_distribution(x, thin)
        return masked_mean_x_greater_than(posterior_distribution, y)


    def posterior_prob_x_greater_than(self, x, y, thin=1):
        """
            P(f(x) > y | Data)
        
            Posterior probability that f(x) is greater than y.
        """
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        probs = []
        samples = self.get_burned_in_samples()
        for theta in samples[::thin]:
            probs.append(self.prob_x_greater_than(x, y, theta))

        return np.ma.masked_invalid(probs).mean()

    def posterior_log_likelihoods(self, x, y):
        #DEPRECATED!
        samples = self.get_burned_in_samples()
        log_likelihoods = []
        for theta in samples:
            params, sigma = self.split_theta(theta)
            log_likelihood = self.ln_likelihood(theta, x, y)
            #TODO: rather add a -np.inf?
            if not np.isnan(log_likelihood) and np.isfinite(log_likelihood):
                log_likelihoods.append(log_likelihood)
        return log_likelihoods

    def mean_posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        return np.ma.masked_invalid(self.posterior_log_likelihoods(x, y)).mean()

    def median_posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        masked_x = np.ma.masked_invalid(self.posterior_log_likelihoods(x, y))
        return np.ma.extras.median(masked_x)

    def max_posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        return np.ma.masked_invalid(self.posterior_log_likelihoods(x, y)).max()

    def posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        return self.median_posterior_log_likelihood(x, y)

    def predictive_std(self, x, thin=1):
        """
           sqrt(Var[f(x)])
        """
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).std()

    def dic(self, x, y):
        """ Deviance Information Criterion. """
        samples = self.get_burned_in_samples()
        deviances = []
        for theta in samples:
            params, sigma = self.split_theta(theta)
            deviance = -2 * self.ln_likelihood(theta, x, y)
            deviances.append(deviance)
        mean_theta = samples.mean(axis=0)
        theta_mean_deviance = -2 * self.ln_likelihood(mean_theta, x, y)
        DIC = 2 * np.mean(deviances) - theta_mean_deviance
        return DIC


class LinearCurveModel(CurveModel):
    """
        Fits a function f(x) = a * x + b using OLS.
    """

    def __init__(self, *arg, **kwargs):
        if "default_vals" in kwargs:
            logging.warn("default values not needed for the linear model.")
        kwargs["default_vals"] = {"a": 0, "b": 0}
        kwargs["min_vals"] = {"a": 0}
        super(LinearCurveModel, self).__init__(
                function=all_models["linear"],
                *arg,
                **kwargs)

    def fit(self, x, y, weights=None, start_from_default=True):
        return self.fit_ml(x, y, weights)

    def fit_ml(self, x, y, weights):
        """
            Ordinary Least Squares fit.

            TODO: use the weights!
        """
        #TODO: check if the results agree with the minimum/maximum values!
        X = np.asarray([np.ones(len(x)), x]).T
        bh = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
        a = bh[1]
        b = bh[0]
        sigma = (y-self.function(x, a, b)).std()
        self.ml_params = np.asarray([a, b, sigma])
        return True

    def predict(self, x):
        a = self.ml_params[0]
        b = self.ml_params[1]
        return a * x + b


class LinearMCMCCurveModel(MCMCCurveModel):
    def __init__(self, **kwargs):
        ml_curve_model = LinearCurveModel()
        super(LinearMCMCCurveModel, self).__init__(
            function=ml_curve_model.function,
            min_vals=ml_curve_model.min_vals,
            max_vals=ml_curve_model.max_vals,
            default_vals=ml_curve_model.default_vals,
            **kwargs)
        self.ml_curve_model = ml_curve_model


def model_ln_prob(theta, model, x, y):
    return model.ln_prob(theta, x, y)


class MCMCCurveModelCombination(object):

    def __init__(self,
            ml_curve_models,
            xlim,
            burn_in=500,
            nwalkers=100,
            nsamples=2500,
            normalize_weights=True,
            monotonicity_constraint=True,
            soft_monotonicity_constraint=False,
            initial_model_weight_ml_estimate=False,
            normalized_weights_initialization="constant",
            strictly_positive_weights=True,
            sanity_check_prior=True,
            nthreads=1,
            recency_weighting=True):
        """
            xlim: the point on the x axis we eventually want to make predictions for.
        """
        self.ml_curve_models = ml_curve_models
        self.xlim = xlim
        self.burn_in = burn_in
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.normalize_weights = normalize_weights
        assert not (monotonicity_constraint and soft_monotonicity_constraint), "choose either the monotonicity_constraint or the soft_monotonicity_constraint, but not both"
        self.monotonicity_constraint = monotonicity_constraint
        self.soft_monotonicity_constraint = soft_monotonicity_constraint
        self.initial_model_weight_ml_estimate = initial_model_weight_ml_estimate
        self.normalized_weights_initialization = normalized_weights_initialization
        self.strictly_positive_weights = strictly_positive_weights
        self.sanity_check_prior = sanity_check_prior
        self.nthreads = nthreads
        self.recency_weighting = recency_weighting
        #the constant used for initializing the parameters in a ball around the ML parameters
        self.rand_init_ball = 1e-6
        self.name = "model combination"# (%s)" % ", ".join([model.name for model in self.ml_curve_models])

    def fit(self, x, y, model_weights=None):
        if self.fit_ml_individual(x, y, model_weights):
            #run MCMC:
            self.fit_mcmc(x, y)
            return True
        else:
            print("fit_ml_individual failed")
            return False

    def y_lim_sanity_check(self, ylim):
        # just make sure that the prediction is not below 0 nor insanely big
        # HOWEVER: there might be cases where some models might predict value larger than 1.0
        # and this is alright, because in those cases we don't necessarily want to stop a run.
        if not np.isfinite(ylim) or ylim < 0. or ylim > 100.0:
            print('Check for ylim=%d violated.' % (ylim))
            return False
        else:
            return True

    def fit_ml_individual(self, x, y, model_weights):
        """
            Do a ML fit for each model individually and then another ML fit for the combination of models.
        """
        self.fit_models = []
        for model in self.ml_curve_models:
            if model.fit(x, y):
                ylim = model.predict(self.xlim)
                if not self.y_lim_sanity_check(ylim):
                    print("ML fit of model %s is out of bound range [0.0, 100.] at xlim." % (model.function.__name__))
                    continue
                params, sigma = model.split_theta_to_array(model.ml_params)
                if not np.isfinite(self.ln_model_prior(model, params)):
                    print("ML fit of model %s is not supported by prior." % model.function.__name__)
                    continue
                self.fit_models.append(model)
                    
        if len(self.fit_models) == 0:
            return False

        if model_weights is None:
            if self.normalize_weights:
                if self.normalized_weights_initialization == "constant":
                    #initialize with a constant value
                    #we will sample in this unnormalized space and then later normalize
                    model_weights = [10. for model in self.fit_models]
                else:# self.normalized_weights_initialization == "normalized"
                    model_weights = [1./len(self.fit_models) for model in self.fit_models]
            else:
                if self.initial_model_weight_ml_estimate:
                    model_weights = self.get_ml_model_weights(x, y)
                    print(model_weights)
                    non_zero_fit_models = []
                    non_zero_weights = []
                    for w, model in zip(model_weights, self.fit_models):
                        if w > 1e-4:
                            non_zero_fit_models.append(model)
                            non_zero_weights.append(w)
                    self.fit_models = non_zero_fit_models
                    model_weights = non_zero_weights
                else:
                    model_weights = [1./len(self.fit_models) for model in self.fit_models]

        #build joint ml estimated parameter vector
        model_params = []
        all_model_params = []
        for model in self.fit_models:
            params, sigma = model.split_theta_to_array(model.ml_params)
            model_params.append(params)
            all_model_params.extend(params)

        y_predicted = self.predict_given_params(x, model_params, model_weights)
        sigma = (y - y_predicted).std()

        self.ml_params = self.join_theta(all_model_params, sigma, model_weights)
        self.ndim = len(self.ml_params)
        if self.nwalkers < 2*self.ndim:
            self.nwalkers = 2*self.ndim
            print("warning: increasing number of walkers to 2*ndim=%d" % (self.nwalkers))
        return True


    def get_ml_model_weights(self, x, y_target):
        """
            Get the ML estimate of the model weights.
        """

        """
            Take all the models that have been fit using ML.
            For each model we get a prediction of y: y_i

            Now how can we combine those to reduce the squared error:

                argmin_w (y_target - w_1 * y_1 - w_2 * y_2 - w_3 * y_3 ...)

            Deriving and setting to zero we get a linear system of equations that we need to solve.


            Resource on QP:
            http://stats.stackexchange.com/questions/21565/how-do-i-fit-a-constrained-regression-in-r-so-that-coefficients-total-1
            http://maggotroot.blogspot.de/2013/11/constrained-linear-least-squares-in.html
        """
        num_models = len(self.fit_models)
        y_predicted = []
        b = []
        for model in self.fit_models:
            y_model = model.predict(x)
            y_predicted.append(y_model)
            b.append(y_model.dot(y_target))
        a = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(num_models):
                a[i, j] = y_predicted[i].dot(y_predicted[j])
                #if i == j:
                #    a[i, j] -= 0.1 #constraint the weights!
        a_rank = np.linalg.matrix_rank(a)
        if a_rank != num_models:
            print("Rank %d not sufficcient for solving the linear system. %d needed at least." % (a_rank, num_models))
        try:
            print(np.linalg.lstsq(a, b)[0])
            print(np.linalg.solve(a, b))
            print(nnls(a, b)[0])
            ##return np.linalg.solve(a, b)
            weights = nnls(a, b)[0]
            #weights = [w if w > 1e-4 else 1e-4 for w in weights]
            return weights
        #except LinAlgError as e:
        except:
            return [1./len(self.fit_models) for model in self.fit_models]


    #priors
    def ln_prior(self, theta):
        ln = 0
        model_params, sigma, model_weights = self.split_theta(theta)
        for model, params in zip(self.fit_models, model_params):
            ln += self.ln_model_prior(model, params)
        #if self.normalize_weights:
            #when we normalize we expect all weights to be positive
        #we expect all weights to be positive
        if self.strictly_positive_weights and np.any(model_weights < 0):
            return -np.inf
        return ln


    def ln_model_prior(self, model, params):
        if not model.are_params_in_bounds(params):
            print('Model params not in bounds')
            return -np.inf
        if self.monotonicity_constraint:
            #check for monotonicity(this obviously this is a hack, but it works for now):
            x_mon = np.linspace(2, self.xlim, 100)
            y_mon = model.function(x_mon, *params)
            if np.any(np.diff(y_mon) < 0):
                print('Monotonicity violated')
                return -np.inf
        # elif self.soft_monotonicity_constraint:
        #     #soft monotonicity: defined as the last value being bigger than the first one
        #     x_mon = np.asarray([2, self.xlim])
        #     y_mon = model.function(x_mon, *params)
        #     if y_mon[0] > y_mon[-1]:
        #         print('Soft monotonicity violated.')
        #         return -np.inf
        # ylim = model.function(self.xlim, *params)
        # #sanity check for ylim
        # if self.sanity_check_prior and not self.y_lim_sanity_check(ylim):
        #     print('Sanity checks violated.')
        #     return -np.inf
        return 0.0

    #likelihood
    def ln_likelihood(self, theta, x, y):
        y_model, sigma = self.predict_given_theta(x, theta)

        if self.recency_weighting:
            weight = recency_weights(len(y))
            ln_likelihood = (weight*norm.logpdf(y-y_model, loc=0, scale=sigma)).sum()
        else:
            ln_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()

        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood

    def ln_prob(self, theta, x, y):
        """
            posterior probability
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y)

    def split_theta(self, theta):
        """
            theta is structured as follows:
            for each model i
                for each model parameter j
            theta = (theta_ij, sigma, w_i)
        """
        num_models = len(self.fit_models)

        model_weights = theta[-len(self.fit_models):]

        all_model_params = []
        for model in self.fit_models:
            num_model_params = len(model.function_params)
            model_params = theta[:num_model_params]
            all_model_params.append(model_params)

            theta = theta[num_model_params:]
        sigma = theta[0]
        model_weights = theta[1:]
        assert len(model_weights) == len(self.fit_models)
        return all_model_params, sigma, model_weights


    def join_theta(self, model_params, sigma, model_weights):
        #assert len(model_params) == len(model_weights)
        theta = []
        theta.extend(model_params)
        theta.append(sigma)
        theta.extend(model_weights)
        return theta

    def fit_mcmc(self, x, y):
        #initialize in an area around the starting position

        assert self.ml_params is not None
        pos = [self.ml_params + self.rand_init_ball*np.random.randn(self.ndim) for i in range(self.nwalkers)]

        if self.nthreads <= 1:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                self.ndim,
                self.ln_prob,
                args=(x, y))
        else:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                model_ln_prob,
                args=(self, x, y),
                threads=self.nthreads)
        sampler.run_mcmc(pos, self.nsamples)
        self.mcmc_chain = sampler.chain

        if self.normalize_weights:
            self.normalize_chain_model_weights()

    def normalize_chain_model_weights(self):
        """
            In the chain we sample w_1,... w_i however we are interested in the model
            probabilities p_1,... p_i
        """
        model_weights_chain = self.mcmc_chain[:,:,-len(self.fit_models):]
        model_probabilities_chain = model_weights_chain / model_weights_chain.sum(axis=2)[:,:,np.newaxis]
        #replace in chain
        self.mcmc_chain[:,:,-len(self.fit_models):] = model_probabilities_chain

    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def print_probs(self):
        burned_in_chain = self.get_burned_in_samples()
        model_probabilities = burned_in_chain[:,-len(self.fit_models):]
        print(model_probabilities.mean(axis=0))

    def predict_given_theta(self, x, theta):
        """
            returns y_predicted, sigma
        """
        model_params, sigma, model_weights = self.split_theta(theta)

        y_predicted = self.predict_given_params(x, model_params, model_weights)

        return y_predicted, sigma

    def predict_given_params(self, x, model_params, model_weights):
        """
            returns y_predicted
        """
        if self.normalize_weights:
            model_weight_sum = np.sum(model_weights)
            model_ws = [weight/model_weight_sum for weight in model_weights]
        else:
            model_ws = model_weights

        y_model = []
        for model, model_w, params in zip(self.fit_models, model_ws, model_params):
            y_model.append(model_w*model.function(x, *params))
        y_predicted = reduce(lambda a, b: a+b, y_model)
        return y_predicted

    def prob_x_greater_than(self, x, y, theta):
        """
            P(f(x) > y | Data, theta)
        """
        model_params, sigma, model_weights = self.split_theta(theta)

        y_predicted = self.predict_given_params(x, model_params, model_weights)

        cdf = norm.cdf(y, loc=y_predicted, scale=sigma)
        return 1. - cdf

    def posterior_prob_x_greater_than(self, x, y, thin=1):
        """
            P(f(x) > y | Data)

            thin: only use every thin'th sample
        
            Posterior probability that f(x) is greater than y.
        """
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        probs = []
        samples = self.get_burned_in_samples()
        for theta in samples[::thin]:   
            probs.append(self.prob_x_greater_than(x, y, theta))
        return np.ma.masked_invalid(probs).mean()


    def posterior_mean_prob_x_greater_than(self, x, y, thin=1):
        """
            P(E[f(x)] > E[y] | Data)

            thin: only use every thin'th sample
        
            Posterior probability that the expected valuef(x) is greater than 
            the expected value of y.
        """
        posterior_distribution = self.predictive_distribution(x, thin)
        return masked_mean_x_greater_than(posterior_distribution, y)


    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            model_params, sigma, model_weights = self.split_theta(theta)
            y_predicted = self.predict_given_params(x, model_params, model_weights)
            predictions.append(y_predicted)
        return np.asarray(predictions)

    def predictive_ln_prob_distribution(self, x, y, thin=1):
        """
            posterior log p(y|x,D) for each sample
        """
        #assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        ln_probs = []
        for theta in samples[::thin]:
            ln_prob = self.ln_likelihood(theta, x, y)
            ln_probs.append(ln_prob)
        return np.asarray(ln_probs)

    def posterior_ln_prob(self, x, y, thin=10):
        """
            posterior log p(y|x,D)

            1/S sum p(y|D,theta_s)
            equivalent to:
            logsumexp(log p(y|D,theta_s)) - log(S)
        """
        assert not np.isscalar(x)
        assert not np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        ln_probs = self.predictive_ln_prob_distribution(x, y)
        return logsumexp(ln_probs) - np.log(len(ln_probs))

    def predict(self, x, thin=1):
        """
            E[f(x)]
        """
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).mean()

    def predictive_std(self, x, thin=1):
        """
           sqrt(Var[f(x)])
        """
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).std()

    def serialize(self, fname):
        import pickle
        pickle.dump(self, open(fname, "wb"))


class MlCurveMixtureModel(object):
    """
        Maximum Likelihood fit of a convex combination of curve models
        using the Expectation Maxization algorithm.

        http://www.slideshare.net/butest/lecture-18-gaussian-mixture-models-and-expectation-maximization
        http://melodi.ee.washington.edu/people/bilmes/mypapers/em.pdf
        http://www.igi.tugraz.at/lehre/MLA/WS07/chapter9.pdf

        With Dirichlet prior:
            ftp://tlp.limsi.fr/public/map93.pdf

        Finite Mixture Model with Dirichlet Distribution
            http://blog.datumbox.com/finite-mixture-model-based-on-dirichlet-distribution/

        Variational Bayesian Gaussian Mixture Model (VBGMM)
            http://kittipatkampa.wordpress.com/2010/10/14/variational-bayesian-gaussian-mixture-model-vbgmm/
    """
    def __init__(self, ml_curve_models):
        self.ml_curve_models = ml_curve_models

    def fit(self, x, y, num_iter=1):
        fit_models = []
        for model in self.ml_curve_models:
            if model.fit(x, y, start_from_default=True):
                fit_models.append(model)
        model_weights = [1./len(fit_models) for m in fit_models]
        if len(fit_models) == 0:
            return False
        try:
            for i in range(0, num_iter):
                #E-step:
                responsibilities = []
                for model_weight, model in zip(model_weights, fit_models):
                    #responsibilities.append(0.000001 + model_weight * model.likelihood(x, y))
                    #responsibilities.append(0.0001 + model_weight * model.likelihood(x, y))
                    responsibilities.append(0.000001 + model_weight * model.likelihood(x, y))
                responsibilities = np.asarray(responsibilities)
                #normalize:
                responsibilities = responsibilities / responsibilities.sum(axis=0)

                #M-step:
                previous_fit_model_weights = responsibilities.mean(axis=1)
                new_fit_models = []
                model_weights = []
                for model_idx, model in enumerate(fit_models):
                    if (previous_fit_model_weights[model_idx] > 0.000001
                        and model.fit(x, y, responsibilities[model_idx, :], start_from_default=False)):
                        new_fit_models.append(model)
                        #model_weights.append(previous_fit_model_weights[model_idx])
                        model_weights.append(0.01 + previous_fit_model_weights[model_idx])
                model_weights = np.asarray(model_weights)
                #renormalize (in case a model couldn't be fit anymore)
                model_weights = model_weights / model_weights.sum()
                fit_models = new_fit_models
                for model_weight, model in zip(model_weights, fit_models):
                    logging.debug("%s %f" % (model.function.__name__, model_weight))
            
            #print model_weights
            self.model_weights = model_weights
            self.fit_models = fit_models
            return True
        except: 
            return False

    def predict(self, x):
        y_predicted = None
        for model_weight, model in zip(self.model_weights, self.fit_models):
            if y_predicted is None:
                y_predicted = model_weight * model.predict(x)
            else:
                y_predicted += model_weight * model.predict(x)
        return y_predicted


class MCMCCurveMixtureModel(object):

    def __init__(self,
            ml_curve_models,
            xlim,
            burn_in=600,
            nwalkers=80,
            nsamples=5000,
            monotonicity_constraint=True,
            soft_monotonicity_constraint=False,
            nthreads=1,
            recency_weighting=True):
        """
            xlim: the point on the x axis we eventually want to make predictions for.
        """
        self.ml_curve_models = ml_curve_models
        self.ml_curve_mixture_model = MlCurveMixtureModel(ml_curve_models)
        self.xlim = xlim
        self.burn_in = burn_in
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        assert not (monotonicity_constraint and soft_monotonicity_constraint), "choose either the monotonicity_constraint or the soft_monotonicity_constraint, but not both"
        self.monotonicity_constraint = monotonicity_constraint
        self.soft_monotonicity_constraint = soft_monotonicity_constraint
        self.nthreads = nthreads
        self.recency_weighting = recency_weighting
        #the constant used for initializing the parameters in a ball around the ML parameters
        self.rand_init_ball = 1e-6

    def fit(self, x, y):
        if self.fit_ml_individual(x, y):
            #run MCMC:
            self.fit_mcmc(x, y)
            return True
        else:
            print("fit_ml_individual failed")
            return False

    def fit_ml_individual(self, x, y):
        """
            Do a ML fit for each model individually and then another ML fit for the combination of models.
        """
        if self.ml_curve_mixture_model.fit(x, y):
            model_weights = self.ml_curve_mixture_model.model_weights
            self.fit_models = self.ml_curve_mixture_model.fit_models
        else:
            self.fit_models = []
            for model in self.ml_curve_models:
                if model.fit(x, y):
                    if np.isfinite(self.ln_model_prior(model, model.ml_params)):
                        self.fit_models.append(model)
                    else:
                        print("ML fit of model %s is not supported by prior." % model.function.__name__)
            model_weights = [10. for model in self.fit_models]
            if len(self.fit_models) == 0:
                return False

        #build joint ml estimated parameter vector
        all_model_params = []
        for model in self.fit_models:
            all_model_params.extend(model.ml_params)
        print("model weights: ", model_weights)
        self.ml_params = self.join_theta(all_model_params, model_weights)
        self.ndim = len(self.ml_params)
        return True

    #priors
    def ln_prior(self, theta):
        ln = 0
        model_thetas, model_weights = self.split_theta(theta)
        for model, theta in zip(self.fit_models, model_thetas):
            ln += self.ln_model_prior(model, theta)
        #if self.normalize_weights:
            #when we normalize we expect all weights to be positive
        #we expect all weights to be positive
        if np.any(model_weights < 0):
            return -np.inf
        return ln

    def ln_model_prior(self, model, theta):
        if not model.are_params_in_bounds(theta):
            print('Model params not in bounds')
            return -np.inf
        if self.monotonicity_constraint:
            #check for monotonicity(this obviously this is a hack, but it works for now):
            x_mon = np.linspace(2, self.xlim, 100)
            params, sigma = model.split_theta_to_array(theta)
            y_mon = model.function(x_mon, *params)
            if np.any(np.diff(y_mon) < 0):
                print('Monotonicity violated')
                return -np.inf
        elif self.soft_monotonicity_constraint:
            #soft monotonicity: defined as the last value being bigger than the first one
            x_mon = np.asarray([2, self.xlim])
            y_mon = model.function(x_mon, *params)
            if y_mon[0] > y_mon[-1]:
                print('Soft monotonicity violated')
                return -np.inf
        return 0.0

    #likelihood
    def ln_likelihood(self, theta, x, y):
        """
        """
        sample_weights = None
        if self.recency_weighting:
            sample_weights = [10**(1./len(y))] * len(y)
            sample_weights = sample_weights**(np.arange(0, len(y)))

        model_thetas, model_weights = self.split_theta(theta)
        #normalize the weights
        model_weight_sum = np.sum(model_weights)
        model_weights = [weight/model_weight_sum for weight in model_weights]

        ln_likelihoods = []
        for model, model_theta, model_weight in zip(self.fit_models, model_thetas, model_weights):
            ln_likelihood = np.log(model_weight)
            params, sigma = model.split_theta_to_array(model_theta)
            y_model = model.function(x, *params)
            if sample_weights is None:
                ln_likelihood += norm.logpdf(y-y_model, loc=0, scale=sigma).sum()
            else:
                ln_likelihood += (sample_weights*norm.logpdf(y-y_model, loc=0, scale=sigma)).sum()

            ln_likelihoods.append(ln_likelihood)

        if np.any(np.isnan(ln_likelihoods)):
            return -np.inf
        else:
            return logsumexp(ln_likelihoods)

    def ln_prob(self, theta, x, y):
        """
            posterior probability
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y)

    def split_theta(self, theta):
        """
            theta is structured as follows:
            for each model i
                for each model parameter j
            theta = (theta_ij, sigma, w_i)
        """
        num_models = len(self.fit_models)

        model_weights = theta[-len(self.fit_models):]

        all_model_params = []
        for model in self.fit_models:
            num_model_params = len(model.all_param_names)
            model_params = theta[:num_model_params]
            all_model_params.append(model_params)

            theta = theta[num_model_params:]
        model_weights = theta
        assert len(model_weights) == len(self.fit_models)
        return all_model_params, model_weights


    def join_theta(self, model_params, model_weights):
        #assert len(model_params) == len(model_weights)
        theta = []
        theta.extend(model_params)
        theta.extend(model_weights)
        return theta

    def fit_mcmc(self, x, y):
        #initialize in an area around the starting position

        assert self.ml_params is not None
        pos = [self.ml_params + self.rand_init_ball*np.random.randn(self.ndim) for i in range(self.nwalkers)]

        if self.nthreads <= 1:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                self.ndim,
                self.ln_prob,
                args=(x, y))
        else:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                model_ln_prob,
                args=(self, x, y),
                threads=self.nthreads)
        sampler.run_mcmc(pos, self.nsamples)
        self.mcmc_chain = sampler.chain

        #we normalize the weights in the chain model, so that the trace plot make more sense
        self.normalize_chain_model_weights()

    def normalize_chain_model_weights(self):
        """
            In the chain we sample w_1,... w_i however we are interested in the model
            probabilities p_1,... p_i
        """
        model_weights_chain = self.mcmc_chain[:,:,-len(self.fit_models):]
        model_probabilities_chain = model_weights_chain / model_weights_chain.sum(axis=2)[:,:,np.newaxis]
        #replace in chain
        self.mcmc_chain[:,:,-len(self.fit_models):] = model_probabilities_chain

    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def print_probs(self):
        burned_in_chain = self.get_burned_in_samples()
        model_probabilities = burned_in_chain[:,-len(self.fit_models):]
        print(model_probabilities.mean(axis=0))

    def predict_given_theta(self, x, theta):
        """
            returns y_predicted, sigma
        """
        model_params, model_weights = self.split_theta(theta)

        y_predicted = self.predict_given_params(x, model_params, model_weights)

        return y_predicted

    def predict_given_params(self, x, model_thetas, model_weights):
        """
            returns y_predicted
        """
        #normalize the weights
        model_weight_sum = np.sum(model_weights)
        model_ws = [weight/model_weight_sum for weight in model_weights]

        y_model = []
        for model, model_w, theta in zip(self.fit_models, model_ws, model_thetas):
            params, sigma = model.split_theta_to_array(theta)
            y_model.append(model_w*model.function(x, *params))
        y_predicted = reduce(lambda a, b: a+b, y_model)
        return y_predicted

    def prob_x_greater_than(self, x, y, theta):
        """
            P(f(x) > y | Data, theta)
        """
        model_params, model_weights = self.split_theta(theta)

        y_predicted = self.predict_given_params(x, model_params, model_weights)

        cdf = norm.cdf(y, loc=y_predicted, scale=sigma)
        return 1. - cdf

    def posterior_prob_x_greater_than(self, x, y, thin=1):
        """
            P(f(x) > y | Data)

            thin: only use every thin'th sample
        
            Posterior probability that f(x) is greater than y.
        """
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        probs = []
        samples = self.get_burned_in_samples()
        for theta in samples[::thin]:   
            probs.append(self.prob_x_greater_than(x, y, theta))

        return np.ma.masked_invalid(probs).mean()

    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            model_params, sigma, model_weights = self.split_theta(theta)
            y_predicted = self.predict_given_params(x, model_params, model_weights)
            predictions.append(y_predicted)
        return np.asarray(predictions)

    def predict(self, x, thin=1):
        """
            E[f(x)]
        """
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).mean()

    def predictive_std(self, x, thin=1):
        """
            sqrt(Var[f(x)])
        """
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).std()

    def serialize(self, fname):
        import pickle
        pickle.dump(self, open(fname, "wb"))