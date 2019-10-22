from pylab import *
import triangle
import random


def greek_label_mapping(oldlabels):
    labels = []
    for param_name in oldlabels:
        if param_name in ["alpha", "beta", "delta", "sigma"]:#"kappa", 
            labels.append("$\%s$" % param_name)
        else:
            labels.append("$%s$" % param_name)
    return labels

class MCMCCurveModelPlotter(object):
    def __init__(self, model):
        self.model = model
        
    def trace_plot(self,rasterized=False):
        num_plots = len(self.model.all_param_names)
        
        fig, axes = subplots(num_plots, 1, sharex=True, figsize=(8, 9))
        
        for idx, param_name in enumerate(self.model.all_param_names):
        
            axes[idx].plot(self.model.mcmc_chain[:, :, idx].T, color="k",
                alpha=0.4, rasterized=rasterized)
            axes[idx].yaxis.set_major_locator(MaxNLocator(5))
            #axes[0].axhline(m_true, color="#888888", lw=2)
            axes[idx].set_ylabel("$%s$" % param_name)
            
            if idx == num_plots-1:
                axes[idx].set_xlabel("step number")
        
        tight_layout(h_pad=0.0)
        

    def triangle_plot(self, labels=None, truths=None):
        samples = self.model.get_burned_in_samples()
        if labels is None:
            labels = greek_label_mapping(self.model.all_param_names)
        fig = triangle.corner(samples,
            labels=labels,
            truths=truths)

    def posterior_sample_plot(self, x, y=None, vline=None,
            xaxislabel="$x$", yaxislabel="$y$", alpha=0.3,
            label="", color="k", x_axis_scale=0.1, nsamples=50,
            plot_ml_estimate=False, ml_estimate_color="#4682b4",
            rasterized=False):
        samples = self.model.get_burned_in_samples()

        x_plot = x_axis_scale*np.asarray(x)

        if y is not None:
            plot(x_plot, y, color="r", lw=2, alpha=0.8, rasterized=rasterized)

        # Plot some samples onto the data.
        for idx, theta in enumerate(samples[np.random.randint(len(samples), size=nsamples)]):
        #for idx, theta in enumerate(samples):
            #print theta
            predictive_mu, sigma = self.model.predict_given_theta(x, theta)

            if idx == 0:
                plot(x_plot, predictive_mu, color=color, alpha=alpha, label=label, rasterized=rasterized)
            else:
                plot(x_plot, predictive_mu, color=color, alpha=alpha, rasterized=rasterized)
            fill_between(x_plot, predictive_mu-sigma, predictive_mu+sigma, color=color, alpha=0.01, rasterized=rasterized)

        if plot_ml_estimate:
            ml_theta = self.model.ml_curve_model.ml_params
            predictive_mu, sigma = self.model.predict_given_theta(x, ml_theta)
            plot(x_plot, predictive_mu, alpha=1.0, color=ml_estimate_color, lw=3, rasterized=rasterized)
            fill_between(x_plot, predictive_mu-sigma, predictive_mu+sigma, color=ml_estimate_color, alpha=0.3, rasterized=rasterized)

        
        if vline is not None:
            axvline(x_axis_scale*vline, color="k")
        ylim(0, 1)
        xlabel(xaxislabel)
        ylabel(yaxislabel)   
        tight_layout()



    def predictive_density_plot(self, x):
        x_lin =  linspace(0., 1., 100)
        plot(x_lin, self.model.predictive_density(x, x_lin));



class MCMCCurveModelCombinationPlotter(object):
    def __init__(self, model):
        self.model = model
        self.colors = ['r', 'g', 'b', 'y', 'cyan', 'magenta', 'Chocolate', 'Crimson', 'DeepSkyBlue', 'Khaki']
        
    def trace_plot(self, figsize=None):
        num_plots = self.model.ndim
        
        if figsize is None:
            figsize = (8, num_plots)
        fig, axes = subplots(num_plots, 1, sharex=True, figsize=figsize)

        labels = ["$%s$" % (param_name) for model in self.model.fit_models for param_name in model.function_params]
        titles = ["%s" % (model.name) for model in self.model.fit_models for param_name in model.function_params]
        labels.append("$sigma$")
        titles.append("")
        labels.extend(["$w%d$" % idx for idx in range(len(self.model.fit_models))])
        titles.extend(["%s weight" % model.name for model in self.model.fit_models])

        for idx, label, title in zip(list(range(self.model.ndim)), labels, titles):
            axes[idx].set_title(title)
            axes[idx].set_ylabel(label)
            axes[idx].plot(self.model.mcmc_chain[:, :, idx].T, color="k", alpha=0.4)
            axes[idx].yaxis.set_major_locator(MaxNLocator(5))
            #axes[0].axhline(m_true, color="#888888", lw=2)
            
            if idx == num_plots-1:
                axes[idx].set_xlabel("step number")
        
        tight_layout(h_pad=0.0)
        

    def triangle_plot(self, labels=None):
        samples = self.model.get_burned_in_samples()
        if labels is None:
            labels = ["$%s$" % (param_name) for model in self.model.fit_models for param_name in model.function_params]
            labels.append("$sigma$")
            labels.extend(["$w%d$" % idx for idx in range(len(self.model.fit_models))])
            labels = greek_label_mapping(labels)
        fig = triangle.corner(samples,
            labels=labels)

    def weights_triangle_plot(self, labels=None, thin=1):
        samples = self.model.get_burned_in_samples()
        if labels is None:
            labels = ["w%d" % idx for idx in range(len(self.model.fit_models))]
            #labels = greek_label_mapping(labels)
            print(labels)
        fig = triangle.corner(
            samples[::thin,-len(self.model.fit_models):])#,
            #labels=labels)

    def weights_triangle_plot(self, labels=None, thin=1):
        samples = self.model.get_burned_in_samples()
        if labels is None:
            labels = ["w%d" % idx for idx in range(len(self.model.fit_models))]
            #labels = greek_label_mapping(labels)
            print(labels)
        fig = triangle.corner(
            samples[::thin,-len(self.model.fit_models):])#,
            #labels=labels)

    def posterior_plot(self, *args, **kwargs):
        self.posterior_sample_plot(*args, **kwargs)

    def posterior_sample_plot(self, x, y=None, vline=None, alpha=0.1, label="",
            xaxislabel="$x$", yaxislabel="$y$", color="k", x_axis_scale=0.1,
            x_lim=None, plot_individual=False, y_plot_lw=2,
            rasterized=False):
        samples = self.model.get_burned_in_samples()

        if x_lim is None:
            x_predict = x
        else:
            x_predict = np.arange(1, x_lim, 1)

        x = x_axis_scale*x
        x_plot = x_axis_scale*np.asarray(x_predict)

        # Plot some samples onto the data.
        for idx, theta in enumerate(samples[np.random.randint(len(samples), size=100)]):
            predictive_mu, sigma = self.model.predict_given_theta(x_predict, theta)

            if idx == 0:
                plot(x_plot, predictive_mu, color=color, alpha=alpha, label=label, rasterized=rasterized)
            else:
                plot(x_plot, predictive_mu, color=color, alpha=alpha, rasterized=rasterized)

            fill_between(x_plot, predictive_mu-2*sigma, predictive_mu+2*sigma, color=color,
                rasterized=rasterized, alpha=0.01)
            if not plot_individual:
                continue
            #plot the contributions of the individual models:
            model_params, sigma, model_weights = self.model.split_theta(theta)
            if self.model.normalize_weights:
                model_weight_sum = np.sum(model_weights)
                model_probs = [weight/model_weight_sum for weight in model_weights]
            else:
                model_probs = model_weights
            for fit_model, model_color, model_prob, params in zip(self.model.fit_models, self.colors, model_probs, model_params):
                #if fit_model.function.__name__ != "ilog2":
                #    continue
                try:
                    sub_model_predictive_mu = fit_model.function(x, *params)
                    #plot(x_plot, model_prob * sub_model_predictive_mu, color=model_color, alpha=alpha)
                    plot(x_plot, sub_model_predictive_mu, color=model_color, alpha=alpha)
                except:
                    print(("error with model: ", fit_model.function.__name__))

        if y is not None:
            plot(x, y, color="r", lw=y_plot_lw, alpha=0.8, label="data")
        if vline is not None:
            axvline(x_axis_scale*vline, color="k")
        ylim(0, 1)
        xlabel(xaxislabel)
        ylabel(yaxislabel)   
        tight_layout()

    def ml_single_models_plot(self, x, y, vline=None, x_axis_scale=0.1):
        lin_comb = None
        x_plot = x_axis_scale*np.asarray(x)
        for m in self.model.fit_models:
            plot(x_plot, m.predict(x), alpha=0.7, label=m.function.__name__, lw=2)
            model_weight = 1. / len(self.model.fit_models)
            if lin_comb is None:
                lin_comb = model_weight * m.predict(x)
            else:
                lin_comb += model_weight * m.predict(x)
        plot(x_plot, m.predict(x), alpha=0.7, label="linear combination", lw=2)

        plot(x_plot, y, color="r", lw=2, alpha=0.8)
        if vline is not None:
            axvline(x_axis_scale*vline, color="k")
        ylim(0, 1)
        legend(loc=4)
        xlabel("$x$")
        ylabel("$y$")   
        tight_layout()


    def predictive_density_plot(self, x):
        x_lin =  linspace(0., 1., 100)
        plot(x_lin, self.model.predictive_density(x, x_lin));


class MCMCCurveMixtureModelPlotter(object):
    def __init__(self, model):
        self.model = model
        self.colors = ['r', 'g', 'b', 'y', 'cyan', 'magenta', 'Chocolate', 'Crimson', 'DeepSkyBlue', 'Khaki']
        
    def trace_plot(self, figsize=None):
        num_plots = self.model.ndim
        
        if figsize is None:
            figsize = (8, num_plots)
        fig, axes = subplots(num_plots, 1, sharex=True, figsize=figsize)

        labels = ["$%s$" % (param_name) for model in self.model.fit_models for param_name in model.all_param_names]
        titles = ["%s" % (model.name) for model in self.model.fit_models for param_name in model.all_param_names]
        labels.extend(["$w%d$" % idx for idx in range(len(self.model.fit_models))])
        titles.extend(["%s weight" % model.name for model in self.model.fit_models])

        for idx, label, title in zip(list(range(self.model.ndim)), labels, titles):
            axes[idx].set_title(title)
            axes[idx].set_ylabel(label)
            axes[idx].plot(self.model.mcmc_chain[:, :, idx].T, color="k", alpha=0.4)
            axes[idx].yaxis.set_major_locator(MaxNLocator(5))
            #axes[0].axhline(m_true, color="#888888", lw=2)
            
            if idx == num_plots-1:
                axes[idx].set_xlabel("step number")
        
        tight_layout(h_pad=0.0)
        

    def triangle_plot(self):
        samples = self.model.get_burned_in_samples()
        
        labels = ["$%s$" % param_name  for model in self.model.fit_models for param_name in model.all_param_names]
        labels.extend(["$%s$" for model in self.model.fit_models])
        fig = triangle.corner(samples, labels=labels)


    def posterior_plot(self, *args, **kwargs):
        self.posterior_sample_plot(*args, **kwargs)

    def posterior_sample_plot(self, x, y, vline=None, alpha=0.8, label="", color="k", x_axis_scale=0.1):
        samples = self.model.get_burned_in_samples()

        x_plot = x_axis_scale*np.asarray(x)

        # Plot some samples onto the data.
        for idx, theta in enumerate(samples[np.random.randint(len(samples), size=100)]):
            predictive_mu = self.model.predict_given_theta(x, theta)

            if idx == 0:
                plot(x_plot, predictive_mu, color=color, alpha=alpha, label=label)
            else:
                plot(x_plot, predictive_mu, color=color, alpha=alpha)

        plot(x_plot, y, color="r", lw=2, alpha=0.8)
        if vline is not None:
            axvline(x_axis_scale*vline, color="k")
        ylim(0, 1)
        xlabel("$x$")
        ylabel("$y$")   
        tight_layout()


class EnsemblePlotter(object):

    def __init__(self,
                 ensemble_curve_model,
                 colors=['r', 'g', 'b', 'y', 'cyan', 'magenta', 'Chocolate', 'Crimson', 'DeepSkyBlue', 'Khaki']):
        self.ensemble_curve_model = ensemble_curve_model
        self.colors = colors
        assert len(colors) >= len(ensemble_curve_model.all_models), "Not enough colors for plot all fit models. Supply more colors!"
        #TODO: always use the same color for the same model!
        self.model_colors = {model.name: model_color for model, model_color in zip(ensemble_curve_model.all_models, colors)}

    def posterior_plot(self, x, y, vline=None, num_curves=100, x_label="epochs", y_label="accuracy", x_axis_scale=0.1):
        """
            x, y: data the posterior will be plotted upon.
            num_curves: the number of curves to plot
            x_axis_scale: scale the values on the xaxis (only for plotting but not passed to the function)
        """

        x_plot = x_axis_scale*np.asarray(x)

        for i in range(num_curves):
            predictive_mu, sigma, color = self.get_random_curve(x)

            plot(x_plot, predictive_mu, color=color, alpha=0.1)

            fill_between(x_plot, predictive_mu-2*sigma, predictive_mu+2*sigma, color=color, alpha=0.01)


        plot(x_plot, y, color="k", lw=2, alpha=0.8)
        if vline is not None:
            axvline(x_axis_scale*vline, color="k")
        ylim(0, 1)
        xlabel(x_label)
        ylabel(y_label)
        tight_layout()

    def get_random_curve(self, x):
        raise NotImplementedError("get_random_curve needs to be overriden")


class CurveModelEnsemblePlotter(EnsemblePlotter):

    def __init__(self, *args, **kwargs):
        super(CurveModelEnsemblePlotter, self).__init__(*args, **kwargs)

    def get_random_curve(self, x):
        """
            Sample a single curve under the given ensemble model.
        """
        #sample model:
        model_idx = np.random.multinomial(1, self.ensemble_curve_model.model_probabilities).argmax()
        model = self.ensemble_curve_model.fit_models[model_idx]
        if model.name in self.model_colors:
            model_color = self.model_colors[model.name]
        else:
            print(("No color defined for model %s" % model.name))
        #sample curve:
        model_samples = model.get_burned_in_samples()   
        theta_idx = np.random.randint(0, model_samples.shape[0], 1)[0]
        theta = model_samples[theta_idx, :]

        params, sigma = model.split_theta(theta)
        predictive_mu = model.function(x, **params)
        return predictive_mu, sigma, model_color


class CurveEnsemblePlotter(EnsemblePlotter):
    def __init__(self, *args, **kwargs):
        super(CurveEnsemblePlotter, self).__init__(*args, **kwargs)


    def get_random_curve(self, x):
        #sample model:
        model_idx = np.random.multinomial(1, self.ensemble_curve_model.model_probabilities).argmax()
        model = self.ensemble_curve_model.fit_models[model_idx]
        model_theta_probabilities = self.ensemble_curve_model.model_theta_probabilities[model_idx]

        if model.name in self.model_colors:
            model_color = self.model_colors[model.name]
        else:
            print(("No color defined for model %s" % model.name))

        model_samples = model.get_burned_in_samples()

        theta_idx = np.random.multinomial(1, model_theta_probabilities).argmax()
        theta = model_samples[theta_idx, :]
        params, sigma = model.split_theta(theta)
        predictive_mu = model.function(x, **params)
        return predictive_mu, sigma, model_color
