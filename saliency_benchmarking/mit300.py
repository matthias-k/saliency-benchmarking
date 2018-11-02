from __future__ import division, absolute_import, unicode_literals, print_function

import pysaliency
from pysaliency import BluringSaliencyMapModel
from pysaliency.baseline_utils import BaselineModel

from .models import DensitySaliencyMapModel, LogDensitySaliencyMapModel, EqualizedSaliencyMapModel


class SaliencyMapProvider(object):
    def __init__(self, fixations_per_image, kernel_size, centerbias_model):
        self.fixations_per_image = fixations_per_image
        self.kernel_size = kernel_size
        self.centerbias_model = centerbias_model

    def saliency_map_model_for_AUC(self, model):
        return EqualizedSaliencyMapModel(DensitySaliencyMapModel(model))

    def saliency_map_model_for_sAUC(self, model):
        return EqualizedSaliencyMapModel(
            LogDensitySaliencyMapModel(model) -
            LogDensitySaliencyMapModel(self.centerbias_model)
        )

    def saliency_map_model_for_NSS(self, model):
        return DensitySaliencyMapModel(model)

    def saliency_map_model_for_IG(self, model):
        return DensitySaliencyMapModel(model)

    def saliency_map_model_for_CC(self, model):
        # TODO: Check with Zoya whether kenle size is radius or diameter
        return BluringSaliencyMapModel(
            DensitySaliencyMapModel(model), kernel_size=self.kernel_size, mode='constant')

    def saliency_map_model_for_KLDiv(self, model):
        # TODO: Check with Zoya whether kenle size is radius or diameter
        return BluringSaliencyMapModel(
            DensitySaliencyMapModel(model), kernel_size=self.kernel_size, mode='constant')

    def saliency_map_model_for_SIM(self, model):
        return pysaliency.SIMSaliencyMapModel(
            model,
            kernel_size=self.kernel_size,
            fixation_count=self.fixations_per_image,
            backlook=1,
            min_iter=10,
            learning_rate_decay_scheme='validation_loss',
            verbose=True)


class MIT300(SaliencyMapProvider):
    def __init__(self, dataset_location=None):
        mit1003_stimuli, mit1003_fixations = pysaliency.get_mit1003(location=dataset_location)
        # centerbias parameters fitted with maximum likelihood and
        # leave-one-image-out crossvalidationo on MIT1003
        baseline_log_regularization, baseline_log_bandwidth = -12.72608551, -1.6624376
        centerbias_model = BaselineModel(
            mit1003_stimuli,
            mit1003_fixations,
            bandwidth=10**baseline_log_bandwidth,
            eps=10**baseline_log_regularization)

        # extrapolate fixations per image from MIT1003 dataset
        fixations_per_image = int(
            100.0  # fixations per image on MIT1003
            / 15   # subjects per image on MIT1003
            * 39   # subjects per image on MIT300
        )

		# TODO: the original MIT Saliency Benchmark uses 8 cycles/image for
		# computing gaussian convolutions and does so via the Fourier domain,
		# i.e. with zero-padding the image to be square and then cyclic extension.
		# according to the paper, 8 cycles/image corresponds to 1 dva or about 35pixel
        # and therefore we use a Gaussian convolution with 35 pixels and nearest
		# padding (which shouldn't make a lot of a difference due to the sparse
		# fixations. Still it might be nice to implement exactly the original
		# blurring in the SIM saliency map model.
        super(MIT300, self).__init__(
            fixations_per_image=fixations_per_image,
            kernel_size=35,
            centerbias_model=centerbias_model
        )
