import os

import pandas as pd
import pysaliency

from .constants import DATASET_LOCATION, MIT300_FIXATIONS, CAT2000_FIXATIONS
from .saliency_map_provider import MIT1003 as MIT1003_Provider, MIT300 as MIT300_Provider, CAT2000 as CAT2000_Provider
from .models import MITFixationMap
from . import datasets


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


class Benchmark(object):
    metrics = ['AUC', 'sAUC', 'NSS', 'CC', 'KLDiv', 'SIM', 'IG']

    def __init__(self, stimuli, fixations, saliency_map_provider, remove_doublicates=False,
                 antonio_gaussian=False, empirical_maps=None, cache_empirical_maps=True):
        self.stimuli = stimuli

        if remove_doublicates:
            fixations = remove_doublicate_fixations(fixations)

        self.fixations = fixations
        self.saliency_map_provider = saliency_map_provider
        if empirical_maps is not None:
            self.empirical_maps = empirical_maps
        elif not antonio_gaussian:
            self.empirical_maps = pysaliency.FixationMap(stimuli, fixations, kernel_size=23.99, caching=cache_empirical_maps)
        else:
            self.empirical_maps = MITFixationMap(stimuli, fixations, fc=8, caching=cache_empirical_maps)

    def evaluate_model(self, model):
        return pd.Series({metric_name: self.evaluate_metric(metric_name, model) for metric_name in self.metrics})

    def evaluate_metric(self, metric, model):
        if isinstance(model, pysaliency.Model) and metric.lower() != 'ig':
            model = self.saliency_map_model_for_metric(metric, model)
        elif isinstance(model, pysaliency.SaliencyMapModel) and metric.lower() == 'ig':
            return None
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
            return self.evaluate_SIM(model)
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

    def evaluate_AUC(self, model):
        return model.AUC(self.stimuli, self.fixations, average='image', verbose=True)

    def evaluate_sAUC(self, model):
        return model.AUC(self.stimuli, self.fixations, average='image', verbose=True, nonfixations='shuffled')

    def evaluate_NSS(self, model):
        return model.NSS(self.stimuli, self.fixations, average='image', verbose=True)

    def evaluate_IG(self, model):
        return model.information_gain(self.stimuli, self.fixations, verbose=True)

    def evaluate_CC(self, model):
        return model.CC(self.stimuli, self.empirical_maps, verbose=True)

    def evaluate_KLDiv(self, model):
        # uses same regularization approach as old MIT Benchmark implementation
        return model.image_based_kl_divergence(
            self.stimuli, self.empirical_maps, verbose=True,
            minimum_value=0,
            log_regularization=2.2204e-16,
            quotient_regularization=2.2204e-16
        )

    def evaluate_SIM(self, model):
        return model.SIM(self.stimuli, self.empirical_maps, verbose=True)


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

        super(MIT300, self).__init__(stimuli, fixations, saliency_map_provider, remove_doublicates=remove_doublicates, antonio_gaussian=antonio_gaussian, empirical_maps=empirical_maps)


class MIT300Old(MIT300):
    def __init__(self):
        stimuli = datasets.get_mit300()
        fixation_directory = os.path.dirname(MIT300_FIXATIONS)
        empirical_maps = pysaliency.SaliencyMapModelFromDirectory(stimuli, os.path.join(fixation_directory, 'FIXATIONMAPS'))
        super(MIT300Old, self).__init__(remove_doublicates=True, empirical_maps=empirical_maps)

    def evaluate_AUC(self, model):
        return model.AUC_Judd(self.stimuli, self.fixations, verbose=True)

    def evaluate_model(self, model):
        # The MIT Saliency Benchmark resizes saliency maps that don't
        # have the same size as the image.
        if isinstance(model, pysaliency.SaliencyMapModel):
            model = pysaliency.ResizingSaliencyMapModel(model)
        #else:
        #    raise TypeError("Can only evaluate saliency map models!")
        return super(MIT300Old, self).evaluate_model(model)


class CAT2000(Benchmark):
    def __init__(self, remove_doublicates=False, antonio_gaussian=False, empirical_maps=None):
        stimuli = datasets.get_cat2000_test()
        fixations = pysaliency.read_hdf5(CAT2000_FIXATIONS)
        saliency_map_provider = CAT2000_Provider()

        super(CAT2000, self).__init__(stimuli, fixations, saliency_map_provider, remove_doublicates=remove_doublicates, antonio_gaussian=antonio_gaussian, empirical_maps=empirical_maps, cache_empirical_maps=False)


class CAT2000Old(CAT2000):
    def __init__(self):
        stimuli = datasets.get_cat2000_test()
        fixation_directory = os.path.dirname(CAT2000_FIXATIONS)
        empirical_maps = pysaliency.SaliencyMapModelFromDirectory(stimuli, os.path.join(fixation_directory, 'ALIBORJI/TEST_DATA/FIXATIONMAPS'), caching=False)
        super(CAT2000Old, self).__init__(remove_doublicates=True, empirical_maps=empirical_maps)

    def evaluate_AUC(self, model):
        return model.AUC_Judd(self.stimuli, self.fixations, verbose=True)

    def evaluate_model(self, model):
        if isinstance(model, pysaliency.SaliencyMapModel):
            model = pysaliency.ResizingSaliencyMapModel(model, caching=False)
        return super(CAT2000Old, self).evaluate_model(model)
