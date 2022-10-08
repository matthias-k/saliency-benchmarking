"""
Code for evaluating multiple metrics in parallel.

This is especially relevant for scanpath models, where computing one
conditional prediction can be computationally very expensive and we don't
want to repeat it for every metric.

Code transfered from DeepGazeIII/scanpaths/multi_metric_evaluation.py
"""

from hashlib import sha1

from boltons.cacheutils import LRU, cached
from boltons.iterutils import chunked_iter
import numpy as np
import pysaliency
from pysaliency.roc import general_rocs_per_positive
from pysaliency.numba_utils import auc_for_one_positive
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
from tqdm import tqdm


def _array_hash(data):
    return sha1(np.ascontiguousarray(data)).hexdigest()


def _blur_cache_key(args, kwargs, typed=False):
    data, sigma = args
    # we need to include shape since otherwise e.g, constant saliency maps with different shape but same area will be mixed up
    return data.shape, _array_hash(data), sigma


@cached(LRU(max_size=1000), key=_blur_cache_key)
def blur_saliency_map(saliency_map, blur_radius):
    print(saliency_map.shape, blur_radius)
    return gaussian_filter(saliency_map, [blur_radius, blur_radius], mode='nearest')


def _general_aucs(positives, negatives):
    if len(positives) == 1:
        return np.array([auc_for_one_positive(positives[0], negatives)])
    else:
        return general_rocs_per_positive(positives, negatives)


def AUC(saliency_map, xs, ys, nonfixations='uniform', blur_radius=None, **kwargs):
    assert nonfixations == 'uniform'
    x_int = np.array(xs, dtype=np.int)
    y_int = np.array(ys, dtype=np.int)

    if blur_radius is not None:
        saliency_map = blur_saliency_map(saliency_map, blur_radius)

    positives = np.asarray(saliency_map[y_int, x_int], dtype=np.float)
    negatives = saliency_map.flatten().astype(np.float)

    return _general_aucs(positives, negatives)


def sAUC(saliency_map, xs, ys, nonfixation_provider=None, fixation_index=None, blur_radius=None, **kwargs):
    x_int = np.array(xs, dtype=np.int)
    y_int = np.array(ys, dtype=np.int)

    nonfix_xs, nonfix_ys = nonfixation_provider(fixation_index)

    if blur_radius is not None:
        saliency_map = blur_saliency_map(saliency_map, blur_radius)

    positives = np.asarray(saliency_map[y_int, x_int], dtype=np.float)
    negatives = np.asarray(saliency_map[nonfix_ys, nonfix_xs]).astype(np.float)

    return _general_aucs(positives, negatives)


def NSS(saliency_map, xs, ys, blur_radius=None, **kwargs):
    if blur_radius is not None:
        saliency_map = blur_saliency_map(saliency_map, blur_radius)

    return pysaliency.metrics.NSS(saliency_map, xs, ys)


def AUC_for_logdensity(log_density, *args, **kwargs):
    return AUC(np.exp(log_density), *args, **kwargs)


def NSS_for_logdensity(log_density, *args, **kwargs):
    return NSS(np.exp(log_density), *args, **kwargs)


def LL(log_density, xs, ys, **kwargs):
    log_density_sum = logsumexp(log_density)
    if not -0.001 < log_density_sum < 0.001:
        raise ValueError("Log density not normalized! LogSumExp={}".format(log_density_sum))

    x_int = np.array(xs, dtype=np.int)
    y_int = np.array(ys, dtype=np.int)

    return log_density[y_int, x_int]


def IG(log_density, *args, fixation_index=None, baseline_log_densities=None, **kwargs):
    if baseline_log_densities is not None:
        baseline_values = baseline_log_densities[fixation_index]
    else:
        baseline_values = -np.log(np.prod(log_density.shape))
    return (LL(log_density, *args, **kwargs) - baseline_values) / np.log(2)


def evaluate_metrics_for_saliency_model(model, stimuli, fixations, metrics, batch_size=None, fixation_indices=None, verbose=False):
    saliency_map_metric_functions = {
        'AUC': AUC,
        'NSS': NSS
    }

    if batch_size is None:
        prediction_fn = model.conditional_saliency_map_for_fixation
    else:
        prediction_fn = model.conditional_saliency_maps

    return _evaluate_metrics(
        prediction_fn, stimuli, fixations, metrics, verbose=verbose,
        metric_functions=saliency_map_metric_functions,
        batch_size=batch_size,
        fixation_indices=fixation_indices,
    )


def evaluate_metrics_for_probabilistic_model(model, stimuli, fixations, metrics, batch_size=None, fixation_indices=None, verbose=False):
    probabilistic_metric_functions = {
        'AUC': AUC_for_logdensity,
        'NSS': NSS_for_logdensity,
        'LL': LL,
        'IG': IG,
    }

    if batch_size is None:
        prediction_fn = model.conditional_log_density_for_fixation
    else:
        prediction_fn = model.conditional_log_densities

    return _evaluate_metrics(
        prediction_fn, stimuli, fixations, metrics, verbose=verbose,
        metric_functions=probabilistic_metric_functions,
        batch_size=batch_size,
        fixation_indices=fixation_indices,
    )


def _evaluate_metrics(prediction_fn, stimuli, fixations, metrics, metric_functions, batch_size=None, fixation_indices=None, verbose=False):
    """
    if batchsize is not None, prediction_fn is called with a subset of stimuli and fixations is expected to return
    conditional predictions for each fixation.
    """
    if isinstance(metrics, list):
        metrics = {metric: True for metric in metrics}

    for metric_key in list(metrics):
        if metrics[metric_key] is True:
            metrics[metric_key] = metric_functions[metric_key]

    results = {
        metric_key: [] for metric_key in metrics
    }

    if batch_size is None:
        def prediction_iter():
            for fixation_index in tqdm(range(len(fixations.x)), disable=not verbose):
                conditional_prediction = prediction_fn(stimuli, fixations, fixation_index)
                outside_fixation_index = fixation_indices[fixation_index]
                yield fixation_index, outside_fixation_index, conditional_prediction
    else:
        def prediction_iter():
            for _fixation_indices in chunked_iter(tqdm(range(len(fixations.x)), disable=not verbose), batch_size):
                conditional_predictions = prediction_fn(stimuli, fixations[_fixation_indices])
                outside_fixation_indices = fixation_indices[_fixation_indices]
                yield from zip(_fixation_indices, outside_fixation_indices, conditional_predictions)

    for fixation_index, outside_fixation_index, conditional_prediction in prediction_iter():
        xs = fixations.x[[fixation_index]]
        ys = fixations.y[[fixation_index]]
        for metric_name, metric_function in metrics.items():
            results[metric_name].extend(metric_function(conditional_prediction, xs, ys, fixation_index=outside_fixation_index))

    return results
