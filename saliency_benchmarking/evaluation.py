import pandas as pd
import pysaliency

from .saliency_map_provider import MIT1003 as MIT1003_Provider, MIT300 as MIT300_Provider
from .models import MITFixationMap


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
                 antonio_gaussian=False):
        self.stimuli = stimuli

        if remove_doublicates:
            fixations = remove_doublicate_fixations(fixations)

        self.fixations = fixations
        self.saliency_map_provider = saliency_map_provider
        if not antonio_gaussian:
            self.empirical_maps = pysaliency.FixationMap(stimuli, fixations, kernel_size=23.99)
        else:
            self.empirical_maps = MITFixationMap(stimuli, fixations, fc=8)

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
        return model.image_based_kl_divergence(self.stimuli, self.empirical_maps, verbose=True)

    def evaluate_SIM(self, model):
        return model.SIM(self.stimuli, self.empirical_maps, verbose=True)


class MIT1003(Benchmark):
    def __init__(self, dataset_location):
        stimuli, fixations = pysaliency.get_mit1003(location=dataset_location)
        stimuli = stimuli[:10]
        fixations = fixations[fixations.n < 10]
        saliency_map_provider = MIT1003_Provider(dataset_location)

        super(MIT1003, self).__init__(stimuli, fixations, saliency_map_provider)


class MIT300(Benchmark):
    def __init__(self, dataset_location, fixation_file, remove_doublicates=False, antonio_gaussian=False):
        stimuli = pysaliency.get_mit300(location=dataset_location)
        fixations = pysaliency.read_hdf5(fixation_file)
        saliency_map_provider = MIT300_Provider(dataset_location)

        super(MIT300, self).__init__(stimuli, fixations, saliency_map_provider, remove_doublicates=remove_doublicates, antonio_gaussian=antonio_gaussian)


class MIT300Old(MIT300):
    def __init__(self, dataset_location, fixation_file):
        super(MIT300Old, self).__init__(dataset_location, fixation_file, remove_doublicates=True, antonio_gaussian=True)

    def evaluate_AUC(self, model):
        return model.AUC_Judd(self.stimuli, self.fixations, verbose=True)