from __future__ import division, absolute_import, unicode_literals, print_function

from boltons.cacheutils import LRU
import numpy as np
import pysaliency
from pysaliency import Model, SaliencyMapModel
from pysaliency.datasets import get_image_hash
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import logsumexp
import tensorflow as tf


def tf_logsumexp(data, axis=0):
    with tf.Graph().as_default() as g:
        input_tensor = tf.placeholder(tf.float32, name='input_tensor')
        output_tensor = tf.reduce_logsumexp(input_tensor, axis=axis)

        with tf.Session(graph=g) as sess:
            return sess.run(output_tensor, {input_tensor: data})


class DensitySaliencyMapModel(SaliencyMapModel):
    """Uses fixation density as predicted by a probabilistic model as saliency maps"""
    def __init__(self, parent_model, **kwargs):
        super().__init__(caching=False, **kwargs)
        self.parent_model = parent_model

    def _saliency_map(self, stimulus):
        return np.exp(self.parent_model.log_density(stimulus))


class LogDensitySaliencyMapModel(SaliencyMapModel):
    """Uses fixation log density as predicted by a probabilistic model as saliency maps"""
    def __init__(self, parent_model, **kwargs):
        super().__init__(caching=False, **kwargs)
        self.parent_model = parent_model

    def _saliency_map(self, stimulus):
        return self.parent_model.log_density(stimulus).copy()


class EqualizedSaliencyMapModel(SaliencyMapModel):
    """Equalizes saliency maps to have uniform histogram"""
    def __init__(self, parent_model, **kwargs):
        super(EqualizedSaliencyMapModel, self).__init__(caching=False, **kwargs)
        self.parent_model = parent_model

    def _saliency_map(self, stimulus):
        smap = self.parent_model.saliency_map(stimulus)
        smap = np.argsort(np.argsort(smap.flatten())).reshape(smap.shape)
        smap = smap.astype(float)
        smap /= np.prod(smap.shape)
        return smap


class ShuffledBaselineModel(pysaliency.Model):
    """Predicts a mixture of all predictions for other images"""
    def __init__(self, parent_model, stimuli, resized_predictions_cache_size=5000,
                 compute_size=(500, 500),
                 **kwargs):
        super().__init__(**kwargs)
        self.parent_model = parent_model
        self.stimuli = stimuli
        self.compute_size = compute_size
        self.resized_predictions_cache = LRU(
            max_size=resized_predictions_cache_size,
            on_miss=self._cache_miss
        )

    def _resize_prediction(self, prediction, target_shape):
        if prediction.shape != target_shape:
            x_factor = target_shape[1] / prediction.shape[1]
            y_factor = target_shape[0] / prediction.shape[0]

            prediction = zoom(prediction, [y_factor, x_factor], order=1, mode='nearest')

            prediction -= logsumexp(prediction)

            assert prediction.shape == target_shape

        return prediction

    def _cache_miss(self, key):
        stimulus = self.stimuli[key]
        return self._resize_prediction(self.parent_model.log_density(stimulus), self.compute_size)

    def _log_density(self, stimulus):
        stimulus_id = get_image_hash(stimulus)

        predictions = []
        prediction = None
        count = 0

        target_shape = (stimulus.shape[0], stimulus.shape[1])

        for k, other_stimulus in enumerate((self.stimuli)):
            if other_stimulus.stimulus_id == stimulus_id:
                continue
            other_prediction = self.resized_predictions_cache[k]
            predictions.append(other_prediction)

        predictions = np.array(predictions) - np.log(len(predictions))
        prediction = tf_logsumexp(predictions, axis=0)

        prediction = self._resize_prediction(prediction, target_shape)

        return prediction
