from __future__ import division, absolute_import, unicode_literals, print_function

import pysaliency
from pysaliency import BluringSaliencyMapModel
from pysaliency.models import ShuffledBaselineModel
from pysaliency.baseline_utils import BaselineModel

from . import datasets
from .models import DensitySaliencyMapModel, LogDensitySaliencyMapModel, EqualizedSaliencyMapModel


class SaliencyMapProvider(object):
    def __init__(self, fixations_per_image, kernel_size):
        self.fixations_per_image = fixations_per_image
        self.kernel_size = kernel_size

    def saliency_map_model_for_AUC(self, model):
        return EqualizedSaliencyMapModel(DensitySaliencyMapModel(model))

    def saliency_map_model_for_sAUC(self, model):
        return EqualizedSaliencyMapModel(
            LogDensitySaliencyMapModel(model) -
            LogDensitySaliencyMapModel(self.baseline_model_for_sAUC(model))
        )

    def baseline_model_for_sAUC(self, model):
        raise NotImplementedError()

    def saliency_map_model_for_NSS(self, model):
        return DensitySaliencyMapModel(model)

    def saliency_map_model_for_IG(self, model):
        return DensitySaliencyMapModel(model)

    def saliency_map_model_for_CC(self, model):
        # TODO: Check with Zoya whether kenle size is radius or diameter
        return BluringSaliencyMapModel(
            DensitySaliencyMapModel(model),
            kernel_size=self.kernel_size,
            mode='constant',
            memory_cache_size=2
        )

    def saliency_map_model_for_KLDiv(self, model):
        # TODO: Check with Zoya whether kenle size is radius or diameter
        return BluringSaliencyMapModel(
            DensitySaliencyMapModel(model),
            kernel_size=self.kernel_size,
            mode='constant',
            memory_cache_size=2
        )

    def saliency_map_model_for_SIM(self, model):
        #print("WARNING! USING SIMPLIFIED SIM MODEL!!!!")
        #return BluringSaliencyMapModel(
        #    DensitySaliencyMapModel(model),
        #    kernel_size=self.kernel_size,
        #    mode='constant',
        #    memory_cache_size=2
        #)
        sim_model = pysaliency.SIMSaliencyMapModel(
            model,
            kernel_size=self.kernel_size,
            fixation_count=self.fixations_per_image,
            backlook=1,
            min_iter=10,
            learning_rate_decay_scheme='validation_loss',
            verbose=True,
            memory_cache_size=2)



        return sim_model


class MIT300(SaliencyMapProvider):
    def __init__(self):
        self.stimuli = datasets.get_mit300()

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
            #kernel_size=35,
            kernel_size=24,
        )

    def baseline_model_for_sAUC(self, model):
        return ShuffledBaselineModel(model, self.stimuli)


class CAT2000(SaliencyMapProvider):
    def __init__(self):
        self.stimuli = datasets.get_cat2000_test()

        # extrapolate fixations per image from CAT2000 train dataset
        fixations_per_image = int(
            336.0  # fixations per image on CAT2000 train
            / 18   # subjects per image on CAT2000 train
            * 24   # subjects per image on CAT2000 test
        )

        # 8 cycles/deg corresponds to 45 pix on CAT2000
        # but by comparing the provided empirical saliency maps I found that they use a kernel size of 40.0px
        super(CAT2000, self).__init__(
            fixations_per_image=fixations_per_image,
            #kernel_size=35,
            kernel_size=40,
            #kernel_size=38,
        )

    def baseline_model_for_sAUC(self, model):
        return ShuffledBaselineModel(model, self.stimuli, memory_cache_size=2)


class MIT1003(SaliencyMapProvider):
    def __init__(self):
        self.stimuli, _ = datasets.get_mit1003()

        # extrapolate fixations per image from MIT1003 dataset
        fixations_per_image = int(
            100.0  # fixations per image on MIT1003
        )

        # TODO: the original MIT Saliency Benchmark uses 8 cycles/image for
        # computing gaussian convolutions and does so via the Fourier domain,
        # i.e. with zero-padding the image to be square and then cyclic extension.
        # according to the paper, 8 cycles/image corresponds to 1 dva or about 35pixel
        # and therefore we use a Gaussian convolution with 35 pixels and nearest
        # padding (which shouldn't make a lot of a difference due to the sparse
        # fixations. Still it might be nice to implement exactly the original
        # blurring in the SIM saliency map model.
        super(MIT1003, self).__init__(
            fixations_per_image=fixations_per_image,
            kernel_size=24,
        )

    def baseline_model_for_sAUC(self, model):
        return ShuffledBaselineModel(model, self.stimuli)


class COCO_Freeview(SaliencyMapProvider):
    def __init__(self):
        self.stimuli = datasets.get_coco_freeview_test()

        fixations_per_image = int(
            154.0  # fixations per image on COCO Freeview train
        )

        # 1dva corresponds to 30.0 px (vertically) and 31.1px horizontally
        super().__init__(
            fixations_per_image=fixations_per_image,
            #kernel_size=35,
            kernel_size=30,
            #kernel_size=38,
        )

    def baseline_model_for_sAUC(self, model):
        return ShuffledBaselineModel(model, self.stimuli, memory_cache_size=2)
