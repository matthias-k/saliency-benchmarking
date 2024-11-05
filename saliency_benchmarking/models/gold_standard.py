import os
from pathlib import Path

import numpy as np
import pysaliency
from pysaliency.baseline_utils import KDEGoldModel
from pysaliency.models import ShuffledBaselineModel, ShuffledSimpleBaselineModel
import yaml

from . import LogDensitySaliencyMapModel, EqualizedSaliencyMapModel


class PrecomputedGoldStandardModel(pysaliency.SubjectDependentModel):
    def __init__(self, stimuli, directory, filename_template='subject{subject}.hdf5', check_shape=True, *args, **kwargs):
        directory = Path(directory)
        subject_models = {}
        subject_files = directory.glob('subject[0-9]*.hdf5')
        for subject_filename in subject_files:
            print(subject_filename)
            subject = int(subject_filename.name.split('subject', 1)[1].split('.hdf5')[0])

            subject_model = pysaliency.HDF5Model(stimuli, subject_filename, caching=True, memory_cache_size=1, check_shape=check_shape)
            subject_models[subject] = subject_model
            #s += 1

        if not subject_models:
            raise ValueError("Didn't find any hdf5 files!")

        super().__init__(subject_models, *args, **kwargs)


def get_gold_standard_model_from_directory(directory, type='crossval', grid_spacing=1):
    config_path = os.path.join(directory, 'config.yaml')
    config = yaml.safe_load(open(config_path))
    params = yaml.safe_load(open(os.path.join(directory, 'results.yaml')))['parameters']
    stimuli, fixations = pysaliency.load_dataset_from_config(config['dataset'])

    subject_models, gold_standard_crossval, gold_standard_upper = get_gold_standard_uniform_centerbias_model(stimuli, fixations, config, params, grid_spacing=grid_spacing)

    if type == 'crossval':
        return gold_standard_crossval
    elif type == 'upper':
        return gold_standard_upper
    else:
        raise ValueError('invalid model type', type)


def get_gold_standard_uniform_centerbias_model(stimuli, fixations, config, params, grid_spacing=1):
    params_optim = params
    model_type = config['model_type']
    assert model_type == 'gold_mixture'
    log_bandwidth = params_optim['log_bandwidth']

    _mixture_models = []
    _weights = []

    def get_gold_subject_model(stimuli, fixations, mixture_weights, mixture_models):
        kde_model = KDEGoldModel(stimuli, fixations, bandwidth=10**log_bandwidth, eps=0, keep_aspect=True, caching=False, grid_spacing=grid_spacing)

        _mixture_models = [kde_model] + mixture_models
        weights = [1.0 - np.sum(mixture_weights)] + mixture_weights

        mixture_model = pysaliency.MixtureModel(_mixture_models,
                                                weights=weights,
                                                caching=True, memory_cache_size=4)
        return mixture_model

    subject_models = {}
    for s in range(fixations.subject_count):
        if model_type == 'gold_mixture':
            _mixture_models = []
            _weights = []

            for model_data in config['regularizations']:
                _mixture_models.append(get_model_for_gold_mixture(stimuli, model_data, s))
                _weights.append(10**params_optim['log_{}'.format(model_data['name'])])

        subject_model = get_gold_subject_model(stimuli, fixations[fixations.subjects != s], _weights, _mixture_models)
        subject_models[s] = subject_model

    gold_standard_crossval = pysaliency.SubjectDependentModel(subject_models)
    if model_type == 'gold_mixture':
        _mixture_models = []
        _weights = []

        for model_data in config['regularizations']:
            _mixture_models.append(get_model_for_gold_mixture(stimuli, model_data, None))
            _weights.append(10**params_optim['log_{}'.format(model_data['name'])])

    gold_standard_upper = get_gold_subject_model(stimuli, fixations, _weights, _mixture_models)

    return subject_models, gold_standard_crossval, gold_standard_upper


def get_model_for_gold_mixture(stimuli, model_data, subject):
    if model_data.get('type', 'hdf5') in ['hdf5', 'uniform']:
        return load_model(stimuli, None, model_data)
    elif model_data['type'] == 'subject_dependent':
        if subject is None:
            return pysaliency.HDF5Model(stimuli, os.path.join(model_data['model_directory'], 'mixture.hdf5'), caching=False)
        else:
            return pysaliency.HDF5Model(stimuli, os.path.join(model_data['model_directory'], 'subject{}.hdf5'.format(subject)), caching=False)
    else:
        raise ValueError(model_data)


def load_model(stimuli, fixations, config):
    model_type = config.get('type', 'hdf5')
    if model_type == 'uniform':
        model = pysaliency.UniformModel()
    elif model_type == 'subject_dependent':
        model = _load_subject_model(stimuli, fixations, config['model_directory'])
    elif model_type == 'hdf5':
        model = pysaliency.HDF5Model(stimuli, config['model_file'], caching=False)
    else:
        raise ValueError('Invalid model type', model_type)

    return model


def get_sAUC_gold_standard_model_from_directory(directory, compute_size=(500, 500), type='crossval', grid_spacing=1, simple_baseline_model=False, precomputation_directory=None):
    config_path = os.path.join(directory, 'config.yaml')
    config = yaml.safe_load(open(config_path))
    stimuli, _ = pysaliency.load_dataset_from_config(config['dataset'])


    if precomputation_directory:
        probabilistic_model = PrecomputedGoldStandardModel(
            stimuli,
            precomputation_directory,
        )
    else:
        probabilistic_model = get_gold_standard_model_from_directory(directory, type=type, grid_spacing=grid_spacing)


    if simple_baseline_model:
        baseline_class = ShuffledSimpleBaselineModel

    else:
        baseline_class = ShuffledBaselineModel

    def _get_sAUC_model(model):
        baseline_model = baseline_class(model, stimuli, compute_size=compute_size, memory_cache_size=4, prepopulate_cache=True)
        #print("Getting average prediction")
        #baseline_model.get_average_prediction(verbose=True)
        return EqualizedSaliencyMapModel(
            LogDensitySaliencyMapModel(model) -
            LogDensitySaliencyMapModel(baseline_model),
            memory_cache_size=4,
        )

    if isinstance(probabilistic_model, pysaliency.Model):
        return _get_sAUC_model(probabilistic_model)
    elif isinstance(probabilistic_model, pysaliency.SubjectDependentModel):
        return pysaliency.SubjectDependentSaliencyMapModel({
            k: _get_sAUC_model(model) for k, model in probabilistic_model.subject_models.items()
        })
    else:
        raise TypeError(probabilistic_model)
