import os
import sys

from boltons.fileutils import mkdir_p
import click
import PIL.Image

import pysaliency
from pysaliency import ModelFromDirectory, HDF5Model

from saliency_benchmarking.mit300 import MIT300


DATASET_LOCATION = 'pysaliency_datasets'
OUTPUT_LOCATION = 'output'


def compute_saliency_maps(model, export_location):
    provider = MIT300(location=DATASET_LOCATION)
    stimuli = pysaliency.get_mit300(location=DATASET_LOCATION)
    

    for metric_name, smap_model in [
        ('AUC', provider.get_saliency_map_model_for_AUC(model)),
        ('sAUC', provider.get_saliency_map_model_for_sAUC(model)),
        ('NSS', provider.get_saliency_map_model_for_NSS(model)),
        ('CC', provider.get_saliency_map_model_for_CC(model)),
        ('KLDiv', provider.get_saliency_map_model_for_KLDiv(model)),
        ('SIM', provider.get_saliency_map_model_for_SIM(model)),
        ]:
        print("Handling", metric_name)
        path = os.path.join(export_location, metric_name)
        mkdir_p(path)

        for filename, stimulus in zip(tqdm(stimuli.filenames), stimuli):
            basename = os.path.basename(filename)
            stem, ext = os.path.splitext(filename)
            target_filename = os.path.join(path, stem+'.png')

            if os.path.isfile(target_filename):
                continue

            saliency_map = smap_model.saliency_map(stimulus)
            image = saliency_map_to_image(saliency_map)

            image.save(target_filename)


def saliency_map_to_image(saliency_map):
    minimum_value = saliency_map.min()
    if minimum_value < 0:
        saliency_map = saliency_map - minimum_value

    saliency_map * 255/saliency_map.max()

    image_data = np.round(saliency_map).astype(np.uint8)
    image = PIL.Image.fromarray(image_data)

    return image

def load_model(location):
    stimuli = pysaliency.get_mit300(location=DATASET_LOCATION)
    if os.path.isfile(location):
        return HDF5Model(stimuli, location)
    elif os.path.isdir(location):
        return ModelFromDirectory(stimuli, location)
    
    raise ValueError("Don't know how to handle model location {}".format(location))


@click.command()
@click.option('--output', help='where to store the resulting saliency maps. Default: output/MODELFILENAMESTEM')
@click.argument('model', help='Where the input model is saved. Can be a HDF5 file or a directory of npy/mat files', target='model_location')
def cli(output, model_location):
    model = load_model(location)

    if output is None:
        model_name = os.path.splitext(os.path.basename(model_location))[0]
        output = os.path.join(OUTPUT_LOCATION, model_name)
    
    compute_saliency_maps(model, output)
