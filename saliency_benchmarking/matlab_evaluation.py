from collections import OrderedDict
import os
from tempfile import TemporaryDirectory

from executor import execute
import pandas as pd
import pysaliency
from pysaliency import SaliencyMapModelFromDirectory, ResizingSaliencyMapModel
from pysaliency.utils import get_minimal_unique_filenames
from tqdm import tqdm

from .models import SaliencyMapModelFromArchive

class MIT300Matlab(object):
    """evaluate model with old matlab code"""
    def __init__(self, dataset_location):
        self.dataset_location = dataset_location
        self.stimuli = pysaliency.get_mit300(location=self.dataset_location)

    def evaluate_model(self, model):
        if isinstance(model, ResizingSaliencyMapModel):
            model = model.parent_model

        tmp_root = 'tmp'
        os.makedirs(tmp_root, exist_ok=True)

        with TemporaryDirectory(dir=tmp_root) as temp_dir:
            if isinstance(model, SaliencyMapModelFromDirectory):
                saliency_map_directory = os.path.abspath(model.directory)

                exts = [os.path.splitext(filename) for filename in model.files]

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
