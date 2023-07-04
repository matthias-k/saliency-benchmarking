from collections import OrderedDict
import os
from tempfile import TemporaryDirectory

from executor import execute
from imageio import imwrite
import numpy as np
import pandas as pd
from PIL import Image
from pysaliency import SaliencyMapModelFromDirectory, ResizingSaliencyMapModel, HDF5SaliencyMapModel
from pysaliency.utils import get_minimal_unique_filenames
from tqdm import tqdm

from .models import SaliencyMapModelFromArchive, IgnoreColorChannelSaliencyMapModel
from . import datasets


class MatlabEvaluation(object):
    def __init__(self, stimuli, code_directory):
        self.stimuli = stimuli
        self.code_directory = code_directory

    def evaluate_model(self, model, filename=None, evaluation_config=None):
        while isinstance(model, (ResizingSaliencyMapModel, IgnoreColorChannelSaliencyMapModel)):
            model = model.parent_model

        tmp_root = 'tmp'
        os.makedirs(tmp_root, exist_ok=True)

        with TemporaryDirectory(dir=tmp_root) as temp_dir:
            if isinstance(model, SaliencyMapModelFromDirectory):
                saliency_map_directory = os.path.abspath(model.directory)

                exts = [os.path.splitext(filename)[-1] for filename in model.files]

            elif isinstance(model, SaliencyMapModelFromArchive):
                print("Extracting predictions")
                saliency_map_directory = os.path.abspath(os.path.join(temp_dir, 'saliency_maps'))
                os.makedirs(saliency_map_directory)

                exts = []
                stimuli_filenames = get_minimal_unique_filenames(self.stimuli.filenames)
                for i in tqdm(range(len(self.stimuli))):
                    filename = model.files[i]
                    basename = os.path.basename(filename)
                    ext = os.path.splitext(basename)[-1]

                    if ext.lower() in ['.mat', '.npy']:
                        saliency_map = model.saliency_map(self.stimuli[0])
                        saliency_map = saliency_map - saliency_map.min()
                        saliency_map /= saliency_map.max()
                        saliency_map *= 255
                        saliency_map = saliency_map.astype(np.uint8)
                        image = Image.fromarray(saliency_map)
                        target_filename = os.path.splitext(stimuli_filenames[i])[0] + '.png'
                        target_filename = os.path.join(saliency_map_directory, target_filename)
                        print(filename, target_filename)
                        os.makedirs(os.path.dirname(target_filename), exist_ok=True)
                        image.save(target_filename)
                        exts.append('.png')
                    else:
                        target_filename = os.path.splitext(stimuli_filenames[i])[0] + ext
                        target_filename = os.path.join(saliency_map_directory, target_filename)
                        print(filename, target_filename)
                        os.makedirs(os.path.dirname(target_filename), exist_ok=True)
                        with open(target_filename, 'wb') as out_file:
                            out_file.write(model.archive.open(filename).read())
                        # check for three channels
                        image = Image.open(target_filename)
                        if np.array(image).ndim == 3:
                            print("Converting to grayscale")
                            image.convert('L').save(target_filename)
                        exts.append(ext)
            elif isinstance(model, HDF5SaliencyMapModel):
                print("Saving predictions to images")
                saliency_map_directory = os.path.abspath(os.path.join(temp_dir, 'saliency_maps'))
                #os.makedirs(saliency_map_directory)

                stimuli_filenames = get_minimal_unique_filenames(self.stimuli.filenames)

                for i in tqdm(range(len(self.stimuli))):
                    saliency_map = model.saliency_map(self.stimuli[i])

                    if saliency_map.dtype in [np.float, np.float32, np.float64, float]:
                        saliency_map -= saliency_map.min()
                        saliency_map /= saliency_map.max()
                        saliency_map *= 255
                        saliency_map = saliency_map.astype(np.uint8)

                    filename = stimuli_filenames[i]
                    #basename = os.path.basename(filename)
                    stem = os.path.splitext(filename)[0]

                    target_filename = os.path.join(saliency_map_directory, stem + '.png')
                    os.makedirs(os.path.dirname(target_filename), exist_ok=True)
                    imwrite(target_filename, saliency_map)
                exts = ['.png']
            else:
                raise TypeError("Can't evaluate model of type {} with matlab".format(type(model)))

            if len(set(exts)) > 1:
                raise ValueError("Matlab cannot handle submissions with different filetypes: {}".format(set(exts)))
            ext = exts[0].split('.')[-1]

            results_dir = os.path.abspath(os.path.join(temp_dir, 'results'))
            os.makedirs(results_dir)

            evaluation_command = f'TestNewModels(\'{saliency_map_directory}\', \'{results_dir}\', [], [], [], \'{ext}\')'
            evaluation_command = f'try, {evaluation_command}, catch me, fprintf(\'%s / %s\\n\',me.identifier,me.message), exit(1), end, exit'

            command = (
                f'matlab'
                + ' -nodisplay'
                + ' -nosplash'
                + ' -nodesktop'
                + ' -r'
                + f' "{evaluation_command}"'
            )
            print(command)

            execute(command, directory=self.code_directory)

            with open(os.path.join(results_dir, 'results.txt')) as f:
                results_txt = f.read()

            return self.extract_results(results_txt)

    def extract_results(self, results_str):
        results_str = 'InfoGain' + results_str.split('\nInfoGain', 1)[1].split('\n\n', 1)[0]
        results_dict = OrderedDict([item.split(':') for item in results_str.split('\n')])

        return pd.Series(results_dict), {key: None for key in results_dict}


class MIT300Matlab(MatlabEvaluation):
    """evaluate model with old matlab code"""

    def __init__(self):
        super().__init__(stimuli=datasets.get_mit300(), code_directory='mit_eval_code')


class CAT2000Matlab(MatlabEvaluation):
    """evaluate model with old matlab code"""

    def __init__(self):
        super().__init__(stimuli=datasets.get_cat2000_test(), code_directory='CAT2000/ALIBORJI/code_forBenchmark_nohist')

    def extract_results_for_category(self, results_str, pattern):
        results_str = results_str.split(pattern, 1)[1].split('\n', 1)[1]
        results_str = results_str.split('\n\n', 1)[0]

        results_dict = OrderedDict([item.split(':') for item in results_str.split('\n')])

        # make names consistent with MIT300
        replace_names = {
            "AUC-Judd metric": "AUC (Judd) metric",
            "SIM metric": "Similarity metric",
            "AUC-Borji metric": "AUC (Borji) metric",
            "sAUC metric": "shuffled AUC metric",
            "CC metric": "Cross-correlation metric",
            "NSS metric": "Normalized Scanpath Saliency metric",
            "EMD metric": "Earth Mover Distance metric",
        }

        results_dict = OrderedDict([(replace_names.get(key, key), value) for key, value in results_dict.items()])

        return results_dict

    def extract_results(self, results_str):
        # TODO: extract results per category, put them into npz
        results_overall = self.extract_results_for_category(results_str, pattern='Overall')

        categories = [
            'Action',
            'Affective',
            'Art',
            'BlackWhite',
            'Cartoon',
            'Fractal',
            'Indoor',
            'Inverted',
            'Jumbled',
            'LineDrawing',
            'LowResolution',
            'Noisy',
            'Object',
            'OutdoorManMade',
            'OutdoorNatural',
            'Pattern',
            'Random',
            'Satelite',
            'Sketch',
            'Social',
        ]

        scores_per_category = {
            "AUC (Judd) metric": [],
            "Similarity metric": [],
            "AUC (Borji) metric": [],
            "shuffled AUC metric": [],
            "Cross-correlation metric": [],
            "Normalized Scanpath Saliency metric": [],
            "Earth Mover Distance metric": [],
            "KL metric": [],
        }

        for category_name in categories:
            pattern = f'Stimuli category {category_name}'
            scores_for_this_category = self.extract_results_for_category(results_str, pattern=pattern)
            for key, value in scores_for_this_category.items():
                scores_per_category[key].append(value)

        return pd.Series(results_overall), scores_per_category
