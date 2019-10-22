from curvefunctions import all_models, curve_combination_models, curve_ensemble_models, model_defaults
from curvemodels import MCMCCurveModel, LinearMCMCCurveModel
from curvemodels import MLCurveModel, LinearCurveModel
from curvemodels import  MCMCCurveModelCombination


def setup_model_combination(xlim,
        models=curve_combination_models,
		recency_weighting=False,
		normalize_weights=True,
		monotonicity_constraint=False,
		soft_monotonicity_constraint=True,
		nthreads=1):

    curve_models = []
    for model_name in models:
        if model_name == "linear":
            m = LinearCurveModel()
        else:
            if model_name in model_defaults:
                m = MLCurveModel(function=all_models[model_name],
                  default_vals=model_defaults[model_name],
                  recency_weighting=recency_weighting)
            else:
                m = MLCurveModel(function=all_models[model_name],
                  recency_weighting=recency_weighting)
        curve_models.append(m)

    model_combination = MCMCCurveModelCombination(curve_models,
        xlim=xlim,
        recency_weighting=recency_weighting,
        normalize_weights=normalize_weights,
        monotonicity_constraint=monotonicity_constraint,
        soft_monotonicity_constraint=soft_monotonicity_constraint,
        nthreads=nthreads)
    return model_combination


pool = None
def setup_curve_model(ensemble_type="curve_model", recency_weighting=False, pool_size=16):
    """
        type: either curve_model or curve
    """
    if pool_size > 1:
        pool = Pool(pool_size)
        map_fun = pool.map
    else:
        map_fun = map

    ensemble_models = []
    for model_name in curve_ensemble_models:
        if model_name == "linear":
            m = LinearCurveModel()
        else:
            if model_name in model_defaults:
                m = MLCurveModel(function=all_models[model_name],
                  default_vals=model_defaults[model_name],
                  recency_weighting=recency_weighting)
            else:
                m = MLCurveModel(function=all_models[model_name],
                  recency_weighting=recency_weighting)
        ensemble_models.append(m)

    if ensemble_type == "curve_model":
        ensemble_curve_model = CurveModelEnsemble(ensemble_models, map=map_fun)
    elif ensemble_type == "curve":
        ensemble_curve_model = CurveEnsemble(ensemble_models, map=map_fun)
    else:
        assert False, "unkown ensemble type"
    
    return ensemble_curve_model


def create_model(model_type, xlim, nthreads=1, recency_weighting=False):
	"""
		type: either curve_combination, curve_model or curve
				curve_combination: Bayesian Model curve_combination
				curve_model: Bayesian Model Averaging
		xlim: the target point that we want to predict eventually
		nthreads: 1 for no parallelization
	"""
	if model_type == "curve_combination":
		curve_model = setup_model_combination(
            xlim=xlim,
            recency_weighting=recency_weighting,
            nthreads=nthreads)
	elif model_type == "curve_model" or model_type == "curve":
		curve_model = setup_curve_model(
            ensemble_type=model_type,
            recency_weighting=recency_weighting,
            pool_size=nthreads)
	return curve_model
