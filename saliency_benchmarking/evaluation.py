from collections.abc import MutableMapping
from functools import partial
import os
import pathlib
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pysaliency
from pysaliency.utils import average_values
from tqdm import tqdm

from .constants import MIT300_FIXATIONS, CAT2000_FIXATIONS, MIT300_BASELINE, CAT2000_BASELINE, COCO_FREEVIEW_FIXATIONS, COCO_FREEVIEW_BASELINE
from .saliency_map_provider import MIT1003 as MIT1003_Provider, MIT300 as MIT300_Provider, CAT2000 as CAT2000_Provider, COCO_Freeview as COCO_Freeview_Provider
from .models import MITFixationMap
from . import datasets
from . import multi_metric_evaluation
from .saving import DistributedNPZSaver


def remove_doublicate_fixations(fixations):
    indices = []
    seen = set()
    for i, (x, y, n) in enumerate(zip(
        fixations.x_int,
        fixations.y_int,
        fixations.n
    )):
        if (x, y, n) not in seen:
            seen.add((x, y, n))
            indices.append(i)

    return fixations[indices]


def print_result(name):
    def decorator(f):
        def wrap(*args, **kwargs):
            result_average, results_full = f(*args, **kwargs)
            print(name, result_average)
            return result_average, results_full
        return wrap
    return decorator


class Benchmark(object):
    metrics = ['IG', 'AUC', 'sAUC', 'NSS', 'CC', 'KLDiv', 'SIM']

    def __init__(self, stimuli, fixations, saliency_map_provider, remove_doublicates=False,
                 antonio_gaussian=False,
                 empirical_maps=None,
                 cache_empirical_maps=True,
                 baseline_model=None,
                 empirical_kernel_size=23.99,
                 empirical_fc=8,
    ):
        self.stimuli = stimuli

        if remove_doublicates:
            fixations = remove_doublicate_fixations(fixations)

        self.fixations = fixations
        self.saliency_map_provider = saliency_map_provider
        if empirical_maps is not None:
            self.empirical_maps = empirical_maps
        elif not antonio_gaussian:
            self.empirical_maps = pysaliency.FixationMap(stimuli, fixations, kernel_size=empirical_kernel_size, caching=cache_empirical_maps)
        else:
            self.empirical_maps = MITFixationMap(stimuli, fixations, fc=empirical_fc, caching=cache_empirical_maps)

        self.baseline_model = baseline_model

    def evaluate_model(self, model, filename=None, evaluation_config=None):
        assert evaluation_config is not None
        print(evaluation_config)

        if not isinstance(model, (pysaliency.Model, pysaliency.SaliencyMapModel)):
            print("Evaluating scanpath model!")
            return _evaluate_model(
                model=model,
                stimuli=self.stimuli,
                fixations=self.fixations,
                baseline_model=self.baseline_model,
                cache_filename=filename,
                random_order=False,
                **evaluation_config,
            )

        cache_data = CacheData(filename)
        average_scores = {}
        full_scores = {}
        for metric_name in self.metrics:
            if f'{metric_name}_average' in cache_data:
                average_score = cache_data[f'{metric_name}_average']
                if np.atleast_1d(average_score)[0] is None:
                    # handle IG for saliency map models
                    average_score = None
                    full_score = None
                else:
                    full_score = cache_data[f'{metric_name}']
            else:
                average_score, full_score = self.evaluate_metric(metric_name, model, cache_filename=filename)
                cache_data[f'{metric_name}_average'] = average_score
                cache_data[metric_name] = full_score

            average_scores[metric_name] = average_score
            if full_score is not None:
                full_scores[metric_name] = full_score

        return pd.Series(average_scores), full_scores
        # return pd.Series({metric_name: self.evaluate_metric(metric_name, model) for metric_name in self.metrics})

    def evaluate_metric(self, metric, model, cache_filename=None):
        if isinstance(model, pysaliency.Model) and metric.lower() != 'ig':
            model = self.saliency_map_model_for_metric(metric, model)
        elif isinstance(model, pysaliency.SaliencyMapModel) and metric.lower() == 'ig':
            return None, None
        if metric.lower() == 'auc':
            return self.evaluate_AUC(model)
        elif metric.lower() == 'sauc':
            return self.evaluate_sAUC(model)
        elif metric.lower() == 'nss':
            return self.evaluate_NSS(model)
        elif metric.lower() == 'ig':
            return self.evaluate_IG(model)
        elif metric.lower() == 'cc':
            return self.evaluate_CC(model)
        elif metric.lower() == 'kldiv':
            return self.evaluate_KLDiv(model)
        elif metric.lower() == 'sim':
            return self.evaluate_SIM(model, cache_filename=cache_filename)
        else:
            raise ValueError(metric)

    def saliency_map_model_for_metric(self, metric, model):
        if metric.lower() == 'auc':
            return self.saliency_map_provider.saliency_map_model_for_AUC(model)
        elif metric.lower() == 'sauc':
            return self.saliency_map_provider.saliency_map_model_for_sAUC(model)
        elif metric.lower() == 'nss':
            return self.saliency_map_provider.saliency_map_model_for_NSS(model)
        elif metric.lower() == 'cc':
            return self.saliency_map_provider.saliency_map_model_for_CC(model)
        elif metric.lower() == 'kldiv':
            return self.saliency_map_provider.saliency_map_model_for_KLDiv(model)
        elif metric.lower() == 'sim':
            return self.saliency_map_provider.saliency_map_model_for_SIM(model)
        else:
            raise ValueError(metric)

    def _average_scores(self, scores):
        return pysaliency.utils.average_values(scores, self.fixations, average='image')

    @print_result('AUC')
    def evaluate_AUC(self, model):
        scores = model.AUCs(self.stimuli, self.fixations, verbose=True)
        return self._average_scores(scores), scores

    @print_result('sAUC')
    def evaluate_sAUC(self, model):
        scores = model.AUCs(self.stimuli, self.fixations, verbose=True, nonfixations='shuffled')
        return self._average_scores(scores), scores

    @print_result('NSS')
    def evaluate_NSS(self, model):
        scores = model.NSSs(self.stimuli, self.fixations, verbose=True)
        return self._average_scores(scores), scores

    @print_result('IG')
    def evaluate_IG(self, model):
        scores = model.information_gains(self.stimuli, self.fixations, verbose=True)
        if len(self.stimuli) == 1000:
            # COCO Freeview, let's do this correctly from the beginning on
            print("averaging IG per image")
            return self._average_scores(scores), scores
        else:
            # TODO: Need to change at some point
            return np.mean(scores), scores

    def evaluate_CC(self, model):
        scores = model.CCs(self.stimuli, self.empirical_maps, verbose=True)
        return np.mean(scores), scores

    def evaluate_KLDiv(self, model):
        # uses same regularization approach as old MIT Benchmark implementation
        scores = model.image_based_kl_divergences(
            self.stimuli, self.empirical_maps, verbose=True,
            minimum_value=0,
            log_regularization=2.2204e-16,
            quotient_regularization=2.2204e-16
        )
        return np.mean(scores), scores

    def evaluate_SIM(self, model, cache_filename=None):
        cache_data = CacheData(cache_filename)

        if 'SIM' in cache_data:
            scores = cache_data['SIM']
        else:
            scores = np.zeros(len(self.stimuli)) * np.nan

        for stimulus_index in tqdm(range(len(self.stimuli))):
            if np.isnan(scores[stimulus_index]):
                scores[stimulus_index] = model.SIM(self.stimuli[[stimulus_index]], self.empirical_maps, verbose=False)
                cache_data['SIM'] = scores

        #scores = model.SIMs(self.stimuli, self.empirical_maps, verbose=True)
        return np.mean(scores), scores


class MIT1003(Benchmark):
    def __init__(self):
        stimuli, fixations = datasets.get_mit1003()
        stimuli = stimuli[:10]
        fixations = fixations[fixations.n < 10]
        saliency_map_provider = MIT1003_Provider()

        super(MIT1003, self).__init__(stimuli, fixations, saliency_map_provider)


class MIT300(Benchmark):
    def __init__(self, remove_doublicates=False, antonio_gaussian=False, empirical_maps=None):
        stimuli = datasets.get_mit300()
        fixations = pysaliency.read_hdf5(MIT300_FIXATIONS)
        saliency_map_provider = MIT300_Provider()
        baseline_model = pysaliency.HDF5Model(stimuli, MIT300_BASELINE)

        super(MIT300, self).__init__(
            stimuli,
            fixations,
            saliency_map_provider,
            remove_doublicates=remove_doublicates,
            antonio_gaussian=antonio_gaussian,
            empirical_maps=empirical_maps,
            baseline_model=baseline_model)


class MIT300Old(MIT300):
    def __init__(self):
        stimuli = datasets.get_mit300()
        fixation_directory = os.path.dirname(MIT300_FIXATIONS)
        empirical_maps = pysaliency.SaliencyMapModelFromDirectory(stimuli, os.path.join(fixation_directory, 'FIXATIONMAPS'))
        super(MIT300Old, self).__init__(remove_doublicates=True, empirical_maps=empirical_maps)

    def evaluate_AUC(self, model):
        return model.AUC_Judd(self.stimuli, self.fixations, verbose=True), None

    def evaluate_model(self, model):
        # The MIT Saliency Benchmark resizes saliency maps that don't
        # have the same size as the image.
        if isinstance(model, pysaliency.SaliencyMapModel):
            model = pysaliency.ResizingSaliencyMapModel(model)
        return super(MIT300Old, self).evaluate_model(model)


class CAT2000(Benchmark):
    def __init__(self, remove_doublicates=False,):
        stimuli = datasets.get_cat2000_test()
        stimuli.cached = False
        fixations = pysaliency.read_hdf5(CAT2000_FIXATIONS)
        saliency_map_provider = CAT2000_Provider()

        baseline_model = pysaliency.HDF5Model(stimuli, CAT2000_BASELINE)

        super(CAT2000, self).__init__(
            stimuli,
            fixations,
            saliency_map_provider,
            remove_doublicates=remove_doublicates,
            antonio_gaussian=False,
            empirical_maps=None,
            cache_empirical_maps=False,
            baseline_model=baseline_model
        )


class COCO_Freeview(Benchmark):
    def __init__(self, remove_doublicates=False):
        stimuli = datasets.get_coco_freeview_test()
        stimuli.cached = False

        fixations = pysaliency.read_hdf5(COCO_FREEVIEW_FIXATIONS)
        fixations = fixations[fixations.lengths > 0]
        fixations = pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations)

        saliency_map_provider = COCO_Freeview_Provider()

        baseline_model = pysaliency.HDF5Model(stimuli, COCO_FREEVIEW_BASELINE)

        super(COCO_Freeview, self).__init__(
            stimuli,
            fixations,
            saliency_map_provider,
            remove_doublicates=remove_doublicates,
            antonio_gaussian=False,
            empirical_kernel_size=30.0,
            empirical_maps=None,
            cache_empirical_maps=False,
            baseline_model=baseline_model
        )


class CAT2000Old(CAT2000):
    def __init__(self):
        stimuli = datasets.get_cat2000_test()
        fixation_directory = os.path.dirname(CAT2000_FIXATIONS)
        empirical_maps = pysaliency.SaliencyMapModelFromDirectory(stimuli, os.path.join(fixation_directory, 'ALIBORJI/TEST_DATA/FIXATIONMAPS'), caching=False)
        super(CAT2000Old, self).__init__(remove_doublicates=True, empirical_maps=empirical_maps)

    def evaluate_AUC(self, model):
        raise NotImplementedError()
        # return model.AUC_Judd(self.stimuli, self.fixations, verbose=True)

    def evaluate_model(self, model):
        if isinstance(model, pysaliency.SaliencyMapModel):
            model = pysaliency.ResizingSaliencyMapModel(model, caching=False)
        return super(CAT2000Old, self).evaluate_model(model)


def _evaluate_model(model, stimuli, fixations, baseline_model, cache_filename, metrics=None, batch_size=100, random_order=True, random_start=True, pixel_per_dva=35):
    if isinstance(model, pysaliency.ScanpathSaliencyMapModel):
        nonfixation_provider = pysaliency.saliency_map_models.FullShuffledNonfixationProvider(stimuli=stimuli, fixations=fixations)
        nonfixation_provider_func = lambda i: nonfixation_provider(stimuli=stimuli, fixations=fixations, i=i)
        #nonfixation_provider_func = partial(nonfixation_provider, stimuli=stimuli, fixations=fixations)
        metrics = {
            'AUC': True,
            'NSS': True,
            'sAUC': partial(multi_metric_evaluation.sAUC, stimuli=stimuli, fixations=fixations, nonfixation_provider=nonfixation_provider_func),
        }

        eval_function = multi_metric_evaluation.evaluate_metrics_for_saliency_model
    elif isinstance(model, pysaliency.ScanpathModel):
        baseline_log_likelihoods = baseline_model.log_likelihoods(stimuli, fixations, verbose=True)
        metrics = {
            'AUC': True,
            'NSS': True,
            # 'sAUC': partial(multi_metric_evaluation.sAUC_for_logdensity, stimuli=stimuli, fixations=fixations, nonfixation_provider=nonfixation_provider)
            'LL': multi_metric_evaluation.IG,
            'IG': partial(multi_metric_evaluation.IG, baseline_log_densities=baseline_log_likelihoods),
        }
        eval_function = multi_metric_evaluation.evaluate_metrics_for_probabilistic_model

    if hasattr(model, 'batch_size'):
        model_batch_size = model.batch_size
    else:
        model_batch_size = None

    metric_scores = {metric_name: np.ones(len(fixations)) * np.nan for metric_name in metrics}

    saver = DistributedNPZSaver(
        filename=cache_filename,
        initial_data=metric_scores,
    )

    data = saver.data
    if not set(data) == set(metrics):
        raise ValueError("Inconsistent metrics in config and data: {} != {}".format(set(data), set(metrics)))

    with tqdm(total=len(saver), initial=saver.done_count) as pbar:
        while saver.done_count < len(saver):
            indices = saver.get_batch(batch_size, random=random_order, random_start=random_start)
            updates = eval_function(model, stimuli, fixations[indices], metrics=metrics, batch_size=model_batch_size, fixation_indices=indices, verbose=True)
            data = saver.update_data(indices, updates)

            descs = []
            for key in ['LL', 'AUC', 'NSS']:
                if key in data:
                    descs.append('{}: {:.04f}'.format(key, average_values(data[key][~np.isnan(data[key])], fixations[~np.isnan(data[key])], average='image')))

            pbar.set_description(' '.join(descs))
            pbar.update(saver.done_count - pbar.n)

    saver.cleanup()

    metric_scores = {}
    for metric, values in data.items():
        score = average_values(values, fixations, average='image')
        metric_scores[metric] = score

    #result_df = pd.DataFrame(metric_scores)
    #print(result_df)
    result_series = pd.Series(metric_scores)
    return result_series, data


class CacheData(MutableMapping):
    def __init__(self, filename) -> None:
        super().__init__()
        self.filename = pathlib.Path(filename)

    @property
    def _data(self):
        if self.filename.is_file():
            with self.filename.open(mode='rb') as f:
                return dict(np.load(f, allow_pickle=True))
        else:
            return {}

#    def __contains__(self, __key: object) -> bool:
#        return __key in self._data

    def __getitem__(self, __key: Any) -> Any:
        print(f"loading {self.filename} for {__key}")
        return self._data[__key]

    def __setitem__(self, __key: Any, __value: Any) -> None:
        print(f"setting {__key} in {self.filename}")
        data = dict(self._data)
        data[__key] = __value
        with self.filename.open(mode='wb') as f:
            np.savez(f, **data)

    def __delitem__(self, __key: Any) -> None:
        print(f"deleting {__key} from {self.filename}")
        data = dict(self._data)
        del data[__key]
        with self.filename.open(mode='wb') as f:
            np.savez(f, **data)

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


