from collections import OrderedDict
import os
from tempfile import TemporaryDirectory

from executor import execute
from imageio import imwrite
import numpy as np
import pandas as pd
import pysaliency
from pysaliency import SaliencyMapModelFromDirectory, ResizingSaliencyMapModel, HDF5SaliencyMapModel
from pysaliency.utils import get_minimal_unique_filenames
from tqdm import tqdm

from .models import SaliencyMapModelFromArchive, IgnoreColorChannelSaliencyMapModel

class MIT300Matlab(object):
    """evaluate model with old matlab code"""
    def __init__(self, dataset_location):
        self.dataset_location = dataset_location
        self.stimuli = pysaliency.get_mit300(location=self.dataset_location)

    def evaluate_model(self, model):
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
                for i in tqdm(range(len(self.stimuli))):
                    filename = model.files[i]
                    basename = os.path.basename(filename)
                    exts.append(os.path.splitext(basename)[-1])
                    target_filename = os.path.join(saliency_map_directory, basename)
                    with open(target_filename, 'wb') as out_file:
                        out_file.write(model.archive.open(filename).read())
            elif isinstance(model, HDF5SaliencyMapModel):
                print("Saving predictions to images")
                saliency_map_directory = os.path.abspath(os.path.join(temp_dir, 'saliency_maps'))
                os.makedirs(saliency_map_directory)

                for i in tqdm(range(len(self.stimuli))):
                    saliency_map = model.saliency_map(self.stimuli[i])

                    if saliency_map.dtype in [np.float, np.float32, np.float64, float]:
                        saliency_map -= saliency_map.min()
                        saliency_map /= saliency_map.max()
                        saliency_map *= 255
                        saliency_map = saliency_map.astype(np.uint8)

                    filename = self.stimuli.filenames[i]
                    basename = os.path.basename(filename)
                    stem = os.path.splitext(basename)[0]

                    target_filename = os.path.join(saliency_map_directory, stem+'.png')
                    imwrite(target_filename, saliency_map)
                exts = ['.png']
            else:
                raise TypeError("Can't evaluate model of type {} with matlab".format(type(model)))

            if len(set(exts)) > 1:
                raise ValueError("Matlab cannot handle submissions with different filetypes: {}".format(set(exts)))
            ext = exts[0].split('.')[-1]

            results_dir = os.path.abspath(os.path.join(temp_dir, 'results'))
            os.makedirs(results_dir)
            
            command = (
                f'matlab'
                + ' -nodisplay'
                + ' -nosplash'
                + ' -nodesktop'
                + ' -r'
                + f' "try, TestNewModels(\'{saliency_map_directory}\', \'{results_dir}\', [], [], [], \'{ext}\'), catch me, fprintf(\'%s / %s\\n\',me.identifier,me.message), end, exit"'
            )
            print(command)

            execute(command, directory='mit_eval_code')

            with open(os.path.join(results_dir, 'results.txt')) as f:
                results_txt = f.read()

            results_str = 'InfoGain' + results_txt.split('\nInfoGain', 1)[1].split('\n\n', 1)[0]
            results_dict = OrderedDict([item.split(':') for item in results_str.split('\n')])

            return pd.Series(results_dict)
