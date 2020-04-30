from __future__ import division, absolute_import, unicode_literals, print_function

import os
import zipfile
import tarfile

from boltons.cacheutils import LRU
from imageio import imread
import numpy as np
import pysaliency
from pysaliency import Model, SaliencyMapModel, FileStimuli
from pysaliency.datasets import get_image_hash
from pysaliency.saliency_map_models import LambdaSaliencyMapModel
from pysaliency.utils import get_minimal_unique_filenames
import rarfile
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import logsumexp
import tensorflow as tf


def tf_logsumexp(data, axis=0):
    with tf.Graph().as_default() as g:
        input_tensor = tf.placeholder(tf.float32, name='input_tensor')
        output_tensor = tf.reduce_logsumexp(input_tensor, axis=axis)

        with tf.Session(graph=g) as sess:
            return sess.run(output_tensor, {input_tensor: data})


def cutoff_frequency_to_gaussian_width(cutoff_frequency, image_size):
    sigma_fourier = cutoff_frequency/np.sqrt(2*np.log(2))

    N = image_size
    sigma_pixel = N/2/np.pi/sigma_fourier

    return sigma_pixel



def antonio_gaussian(img, fc):
    """Port of antonioGaussian.m from the MIT metrics code"""
    sn, sm = np.shape(img)
    n = max(sn, sm)
    n = n + (n % 2)
    n = 2**int(np.ceil(np.log2(n)))
    
    # frequencies:
    fx, fy = np.meshgrid(np.arange(0,n), np.arange(0, n))
    fx = fx-n/2
    fy = fy-n/2

    # convert cut of frequency into gaussian width:
    s=fc/np.sqrt(np.log(2))

    # compute transfer function of gaussian filter:
    gf = np.exp(-(fx**2+fy**2)/(s**2));
    gf = np.fft.fftshift(gf)

    # convolve (in Fourier domain) each color band:
    BF = np.zeros((n,n))
    
    BF = np.real(
        np.fft.ifft2(
            np.fft.fft2(img, [n,n]) * gf
        )
    )
    
    # crop output to have same size than the input
    BF = BF[:sn,:sm]
    
    return BF


class AntonioGaussianMap(pysaliency.saliency_map_models.LambdaSaliencyMapModel):
    def __init__(self, parent_model, fc=6):
        super(AntonioGaussianMap, self).__init__(
            [parent_model],
            lambda s: antonio_gaussian(s[0], fc=fc)
        )
        self.fc = fc


class MITFixationMap(AntonioGaussianMap):
    def __init__(self, stimuli, fixations, fc=8):
        fixation_map = pysaliency.FixationMap(stimuli, fixations)
        super(MITFixationMap, self).__init__(fixation_map, fc=fc)


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


#class SaliencyMapModelFromArchive(SaliencyMapModel):
#    def __init__(self, stimuli, archive_file, **kwargs):
#        if not isinstance(stimuli, FileStimuli):
#            raise TypeError('SaliencyMapModelFromArchive works only with FileStimuli!')
#
#        super(SaliencyMapModelFromArchive, self).__init__(**kwargs)
#        self.stimuli = stimuli
#        self.stimulus_ids = list(stimuli.stimulus_ids)
#        self.archive_file = archive_file
#        _, archive_ext = os.path.splitext(self.archive_file)
#        if archive_ext.lower() == '.zip':
#            self.archive = zipfile.open(self.archive_file)
#        elif archive_ext.lower() == '.tar':
#            self.archive = tarfile.open(self.archive_file)
#        elif archive_ext.lower() == '.rar':
#            self.archive = rarfile.RarFile.open(self.archive_file)
#        else:
#            raise ValueError(archive_ext)
#        
#
#        files = os.listdir(directory)
#        stems = [os.path.splitext(f)[0] for f in files]
#
#        stimuli_files = [os.path.basename(f) for f in stimuli.filenames]
#        stimuli_stems = [os.path.splitext(f)[0] for f in stimuli_files]
#
#        assert set(stimuli_stems).issubset(stems)
#
#        indices = [stems.index(f) for f in stimuli_stems]
#
#        files = [os.path.join(directory, f) for f in files]
#        files = [files[i] for i in indices]
#
#    def _saliency_map(self, stimulus):
#        stimulus_id = get_image_hash(stimulus)
#        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
#        filename = self.files[stimulus_index]
#        return self._load_file(filename)
#
#    def _load_file(self, filename):
#        _, ext = os.path.splitext(filename)
#        if ext.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
#            return imread(filename).astype(float)
#        elif ext.lower() == '.npy':
#            return np.load(filename).astype(float)
#        elif ext.lower() == '.mat':
#            data = loadmat(filename)
#            variables = [v for v in data if not v.startswith('__')]
#            if len(variables) > 1:
#                raise ValueError('{} contains more than one variable: {}'.format(filename, variables))
#            elif len(variables) == 0:
#                raise ValueError('{} contains no data'.format(filename))
#            return data[variables[0]]
#        else:
#            raise ValueError('Unkown file type: {}'.format(ext))
#
#    def can_handle(filename):
#        return zipfile.is_zipfile(filename) or tarfile.is_tarfile(filename) or rarfile.is_rarfile(filename)
#
#    def foobar(self):
#        print("SIP")



class TarFileLikeZipFile(tarfile.TarFile):
    """ Wrapper that makes TarFile behave more like ZipFile """
    def namelist(self):
        filenames = []
        for tar_info in self.getmembers():
            filenames.append(tar_info.name)
        return filenames

    def open(self, name, mode='r'):
        return self.extractfile(name)



#class RarFileLikeZipFile(rarfile.RarFile):
#    """ Wrapper that makes TarFile behave more like ZipFile """
#    def namelist(self):
#        filenames = []
#        for tar_info in self.getmembers():
#            filenames.append(tar_info.name)
#        return filenames
#
#
class SaliencyMapModelFromArchive(SaliencyMapModel):
    def __init__(self, stimuli, archive_file, **kwargs):
        if not isinstance(stimuli, FileStimuli):
            raise TypeError('SaliencyMapModelFromArchive works only with FileStimuli!')

        super(SaliencyMapModelFromArchive, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.stimulus_ids = list(stimuli.stimulus_ids)
        self.archive_file = archive_file
        _, archive_ext = os.path.splitext(self.archive_file)
        if archive_ext.lower() == '.zip':
            self.archive = zipfile.ZipFile(self.archive_file)
        elif archive_ext.lower() == '.tar':
            self.archive = TarFileLikeZipFile(self.archive_file)
        elif archive_ext.lower() == '.rar':
            self.archive = rarfile.RarFile(self.archive_file)
        else:
            raise ValueError(archive_file)
        
        files = self.archive.namelist()
        files = [f for f in files if not '.ds_store' in f.lower()]
        files = [f for f in files if not '__macosx' in f.lower()]
        stems = [os.path.splitext(f)[0] for f in files]

        #stimuli_files = [os.path.basename(f) for f in stimuli.filenames]
        stimuli_files = get_minimal_unique_filenames(stimuli.filenames)
        stimuli_stems = [os.path.splitext(f)[0] for f in stimuli_files]
        
        prediction_filenames = []
        for stimuli_stem in stimuli_stems:
            candidates = [stem for stem in stems if stem.endswith(stimuli_stem)]
            if not candidates:
                raise ValueError("Can't find file for {}".format(stimuli_stem))
            if len(candidates) > 1:
                raise ValueError("Multiple candidates for {}: {}", stimuli_stem, candidates)
            
            target_stem, = candidates
            target_index = stems.index(target_stem)
            target_filename = files[target_index]
            
            prediction_filenames.append(target_filename)
            

        #assert set(stimuli_stems).issubset(stems)

        #indices = [stems.index(f) for f in stimuli_stems]

        #files = [files[i] for i in indices]
        self.files = prediction_filenames

    def _saliency_map(self, stimulus):
        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        filename = self.files[stimulus_index]
        return self._load_file(filename)

    def _load_file(self, filename):
        _, ext = os.path.splitext(filename)
        
        content = self.archive.open(filename)
        
        if ext.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
            return imread(content).astype(float)
        elif ext.lower() == '.npy':
            return np.load(content).astype(float)
        elif ext.lower() == '.mat':
            data = loadmat(content)
            variables = [v for v in data if not v.startswith('__')]
            if len(variables) > 1:
                raise ValueError('{} contains more than one variable: {}'.format(filename, variables))
            elif len(variables) == 0:
                raise ValueError('{} contains no data'.format(filename))
            return data[variables[0]]
        else:
            raise ValueError('Unkown file type: {}'.format(ext))

    @staticmethod
    def can_handle(filename):
        return zipfile.is_zipfile(filename) or tarfile.is_tarfile(filename) or rarfile.is_rarfile(filename)


def _remove_color(saliency_maps):
    saliency_map, = saliency_maps
    if saliency_map.ndim == 3 and saliency_map.shape[-1] == 3:
        color_variance = np.var(saliency_map, axis=-1).max()

        if color_variance > 0:
            raise ValueError("Cannot handle 3-channel saliency maps with other colors than shades of gray")

        return saliency_map[:, :, 0]

    return saliency_map


class IgnoreColorChannelSaliencyMapModel(LambdaSaliencyMapModel):
    def __init__(self, parent_model):
        super().__init__([parent_model], _remove_color, caching=False)
        self.parent_model = parent_model

