from datetime import datetime
from glob import glob
import json
import os
import shutil

from boltons.fileutils import mkdir_p
import click
from executor import execute
from jinja2 import Environment, Markup
import markdown
import numpy as np
import pandas as pd
import PIL.Image
import schema
from scipy.special import logsumexp
from tqdm import tqdm
import yaml

import pysaliency
from pysaliency import ModelFromDirectory, HDF5Model, SaliencyMapModelFromDirectory, HDF5SaliencyMapModel, ResizingSaliencyMapModel

from saliency_benchmarking.constants import DATASET_LOCATION
from saliency_benchmarking.datasets import load_dataset
from saliency_benchmarking.evaluation import MIT300, MIT300Old, MIT1003, CAT2000, CAT2000Old
from saliency_benchmarking.matlab_evaluation import MIT300Matlab, CAT2000Matlab
from saliency_benchmarking.models import ModelFromArchive, SaliencyMapModelFromArchive, IgnoreColorChannelSaliencyMapModel
from saliency_benchmarking.utils import iterate_submissions


def compute_saliency_maps(model, export_location):
    provider = MIT300(dataset_location=DATASET_LOCATION)
    stimuli = pysaliency.get_mit300(location=DATASET_LOCATION)

    for metric_name, smap_model in [
            ('AUC', provider.saliency_map_model_for_AUC(model)),
            ('sAUC', provider.saliency_map_model_for_sAUC(model)),
            ('NSS', provider.saliency_map_model_for_NSS(model)),
            ('IG', provider.saliency_map_model_for_IG(model)),
            ('CC', provider.saliency_map_model_for_CC(model)),
            ('KLDiv', provider.saliency_map_model_for_KLDiv(model)),
            ('SIM', provider.saliency_map_model_for_SIM(model)),
        ]:
        print("Handling", metric_name)
        path = os.path.join(export_location, metric_name)
        mkdir_p(path)

        for filename, stimulus in zip(tqdm(stimuli.filenames), stimuli):
            basename = os.path.basename(filename)
            stem, _ = os.path.splitext(basename)
            target_filename = os.path.join(path, stem + '.png')

            if os.path.isfile(target_filename):
                continue

            saliency_map = smap_model.saliency_map(stimulus)
            image = saliency_map_to_image(saliency_map)

            image.save(target_filename)


def saliency_map_to_image(saliency_map):
    minimum_value = saliency_map.min()
    if minimum_value < 0:
        saliency_map = saliency_map - minimum_value

    saliency_map = saliency_map * 255 / saliency_map.max()

    image_data = np.round(saliency_map).astype(np.uint8)
    image = PIL.Image.fromarray(image_data)

    return image


def load_probabilistic_model(dataset_name, location):
    stimuli = load_dataset(dataset_name)
    if os.path.isfile(location):
        if ModelFromArchive.can_handle(location):
            model = ModelFromArchive(stimuli, location)
        else:
            model = HDF5Model(stimuli, location)
    elif os.path.isdir(location):
        model = ModelFromDirectory(stimuli, location, caching=False)
    else:
        raise ValueError("Don't know how to handle model location {}".format(location))
    print("Testing model")
    for stimulus in stimuli:
        log_density = model.log_density(stimulus)
        log_density_sum = logsumexp(log_density)
        if not -0.001 < log_density_sum < 0.001:
            raise ValueError("Log density not normalized! LogSumExp={}".format(log_density_sum))

    return model


def load_saliency_map_model(dataset_name, location):
    stimuli = load_dataset(dataset_name)
    if os.path.isfile(location):
        if SaliencyMapModelFromArchive.can_handle(location):
            model = SaliencyMapModelFromArchive(stimuli, location)
        else:
            model = HDF5SaliencyMapModel(stimuli, location)
    
    elif os.path.isdir(location):
        model = SaliencyMapModelFromDirectory(stimuli, location, caching=False)
    else:
        raise ValueError("Don't know how to handle model location {}".format(location))

    model = IgnoreColorChannelSaliencyMapModel(model)

    return ResizingSaliencyMapModel(model, caching=False)

@click.group()
def cli():
    pass


@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('-d', '--dataset', type=click.Choice(['MIT300', 'MIT1003']), default='MIT1003', help='Which dataset to evaluate')
@click.option('-t', '--type', type=click.Choice(['saliencymap', 'probabilistic']), default='probabilistic', help='Where the model is a saliency map model or a probilistic model')
@click.option('--output', help='where to store the resulting scores. Default: don\'t store at all')
@click.option('-e', '--evaluation', type=click.Choice(['old-matlab', 'old-python', 'new']), default='new',  help='whether to use the old evaluation schema in matlab or python reimplementation: not more than one fixation per pixel, FFT empirical maps, etc or to use new evaluation scheme')
@click.argument('model-location')
def evaluate_model(dataset, type, output, evaluation, model_location):
    _evaluate_model(dataset, type, output, evaluation, model_location)


def _evaluate_model(dataset, type, output, evaluation, model_location):
    if type.lower() == 'saliencymap':
        model = load_saliency_map_model(dataset, model_location)
    elif type.lower() == 'probabilistic':
        model = load_probabilistic_model(dataset, model_location)

    if dataset.lower() == 'mit300':
        if evaluation == 'old-python':
            benchmark = MIT300Old()
        elif evaluation == 'old-matlab':
            benchmark = MIT300Matlab()
        elif evaluation == 'new':
            benchmark = MIT300()
        else:
            raise ValueError(evaluation)
    elif dataset.lower() == 'cat2000':
        if evaluation == 'old-python':
            benchmark = CAT2000Old()
        elif evaluation == 'old-matlab':
            benchmark = CAT2000Matlab()
        elif evaluation == 'new':
            benchmark = CAT2000()
        else:
            raise ValueError(evaluation)
    elif dataset.lower() == 'mit1003':
        assert evaluation == 'new'
        benchmark = MIT1003()
    else:
        raise ValueError(dataset)

    results = benchmark.evaluate_model(model)
    print(results)

    if output:
        results.to_csv(output, header=False)


def _load_location(location, evaluation='new'):
    config = _load_config(location)
    dataset = config['dataset']
    type = 'probabilistic' if config['model']['probabilistic'] else 'saliencymap'
    model_location=os.path.join(location, config['model']['filename'])

    if type.lower() == 'saliencymap':
        model = load_saliency_map_model(dataset, model_location)
    elif type.lower() == 'probabilistic':
        model = load_probabilistic_model(dataset, model_location)

    if dataset.lower() == 'mit300':
        if evaluation == 'old-python':
            benchmark = MIT300Old()
        elif evaluation == 'old-matlab':
            benchmark = MIT300Matlab()
        elif evaluation == 'new':
            benchmark = MIT300()
        else:
            raise ValueError(evaluation)
    elif dataset.lower() == 'cat2000':
        if evaluation == 'old-python':
            benchmark = CAT2000Old()
        elif evaluation == 'old-matlab':
            benchmark = CAT2000Matlab()
        elif evaluation == 'new':
            benchmark = CAT2000()
        else:
            raise ValueError(evaluation)
    elif dataset.lower() == 'mit1003':
        assert evaluation == 'new'
        benchmark = MIT1003()
    else:
        raise ValueError(dataset)

    return benchmark, model


MaybeString = schema.Or(str, None)

display_schema = schema.Schema({
    schema.Optional('name', default=None): str,
    schema.Optional('published', default=''): str,
    schema.Optional('code', default=''): MaybeString,
    schema.Optional('evaluation_comment', default=''): str,
    schema.Optional('first_tested', default=None): str,
})

config_schema = schema.Schema({
    'model': {
        'name': str,
        'filename': str,
        'probabilistic': bool,
        schema.Optional('loss', default=False): schema.Or('AUC', 'sAUC', 'IG', 'NSS', 'CC', 'KLDiv', 'SIM'),
    },
    schema.Optional('authors'): str,
    schema.Optional('publication'): str,
    schema.Optional('submitted'): str,
    schema.Optional('submitted_first'): str,
    schema.Optional('public', default=False): bool,
    'dataset': schema.Or('MIT300', 'MIT1003', 'CAT2000'),
    schema.Optional('display', default=display_schema.validate({})): display_schema,
})


def check_previous_results(location, accept_results_after, results_directory='results'):
    pattern = os.path.join(location, results_directory, '*.csv')
    for path in reversed(sorted(glob(pattern))):
        filename = os.path.basename(path)
        stem, _ = os.path.splitext(filename)
        if stem >= accept_results_after:
            return path

    return False


def get_result_dates(location, results_directory='results'):
    pattern = os.path.join(location, results_directory, '*.csv')
    dates = []
    for path in reversed(sorted(glob(pattern))):
        filename = os.path.basename(path)
        stem, _ = os.path.splitext(filename)
        stem = stem[:len('YYYY-MM-DD_HH_MM_SS')]
        date = datetime.strptime(stem, '%Y-%m-%d_%H-%M-%S')
        dates.append(date)
    return dates

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--accept-results-after', default='2000')
@click.option('-e', '--evaluation', type=click.Choice(['old-matlab', 'old-python', 'new']), default='new',  help='whether to use the old evaluation schema in matlab or python reimplementation: not more than one fixation per pixel, FFT empirical maps, etc or to use new evaluation scheme')
@click.argument('location')
def evaluate_location(accept_results_after, evaluation, location):
    _evaluate_location(accept_results_after, evaluation, location)


def _load_config(location):
    config_path = os.path.join(location, 'config.yaml')
    if not os.path.isfile(config_path):
        raise ValueError(f'{config_path} does not exist')
    return config_schema.validate(yaml.safe_load(open(config_path)))


def _evaluate_location(accept_results_after, evaluation, location):
    results_directory = 'results'
    if evaluation == 'old-python':
        results_directory += '-old-evaluation'
    elif evaluation == 'old-matlab':
        results_directory += '-old-matlab'

    if accept_results_after:
        previous_results = check_previous_results(location, accept_results_after, results_directory=results_directory)
        if previous_results:
            print(f"Found previous results in {previous_results}")
            return

    config = _load_config(location)
    if config['model']['probabilistic'] and evaluation == 'old-matlab':
        print("Can't evaluate probabilistic models with matlab")
        return

    output_path = os.path.join(
        location,
        results_directory,
        datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S.csv')
    )
    mkdir_p(os.path.dirname(output_path))

    _evaluate_model(
        dataset=config['dataset'],
        model_location=os.path.join(location, config['model']['filename']),
        type='probabilistic' if config['model']['probabilistic'] else 'saliencymap',
        evaluation=evaluation,
        output=output_path,
    )


@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--accept-results-after', default='2000')
@click.option('--submissions-directory', '-d', default='submissions', help='evaluate all submissions in this directory')
def evaluate(accept_results_after, submissions_directory):
    for full_path in iterate_submissions(submissions_directory):
        print("Evaluating", full_path)
        _evaluate_location(accept_results_after, evaluation='new', location=full_path)
        #_evaluate_location(accept_results_after, evaluation='old-python', location=full_path)
        _evaluate_location(accept_results_after, evaluation='old-matlab', location=full_path)


@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--accept-results-after', default='2000')
@click.argument('submission')
def print_results(accept_results_after, submission):
    _print_results(accept_results_after, submission)

def _print_results(accept_results_after, submission):
    config = _load_config(submission)
    results = prepare_results_from_location(submission, only_published=False)
    results_matlab = prepare_results_from_location(submission, only_published=False, results_directory='results-old-matlab')

    data = {
        'results': results,
        'results_matlab': results_matlab,
        'submission': submission,
        'config': config,
    }

    md = markdown.Markdown()
    env = Environment()
    env.filters['markdown'] = lambda text: Markup(md.convert(text))

    def format_encoded_datetime(datetime_str, format_str):
        try:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S')
        return datetime_obj.strftime(format_str)
    env.filters['datetime'] = format_encoded_datetime

    env.trim_blocks = True
    env.lstrip_blocks = True
    template = env.from_string(open('templates/results_mail.jinja2').read())

    results_mail = template.render(**data)
    print(results_mail)


def prepare_results_from_location(location, only_published=True, results_directory='results'):
    previous_results_file = check_previous_results(location, accept_results_after="2000", results_directory=results_directory)
    if not previous_results_file:
        return None
    config = _load_config(location)
    if only_published and not config['public']:
        return None

    results = pd.read_csv(previous_results_file, header=None, index_col=0)[1]
    if 'IG' in results and results['IG']:
        # substract centerbias performance manually for now.
        results['IG'] -= 0.764993252612521
    results['name'] = config['model']['name']
    results.name = config['model']['name']
    return results


@cli.command(context_settings={'help_option_names': ['-h', '--help']}, help="Print leaderboard")
@click.option('--submissions-directory', '-d', default='submissions', help='evaluate all submissions in this directory')
@click.option('--only-public/--not-only-public', '-p', default=True)
@click.option('--dataset', type=click.Choice(['MIT300', 'MIT1003', 'CAT2000']), default='MIT300', help='Which dataset to print leaderboard for')
def leaderboard(submissions_directory, only_public, dataset):
    print(_leaderboard(submissions_directory, only_public, dataset))


def _leaderboard(submissions_directory, only_public, dataset):
    results = []
    for full_path in iterate_submissions(submissions_directory):
        this_results = prepare_results_from_location(full_path, only_published=only_public)
        if this_results is not None:
            this_results['directory'] = full_path
            config = _load_config(full_path)
            if config['dataset'] == dataset:
                results.append(this_results)

    results = pd.DataFrame(results).sort_values('AUC')
    return results



def _website_data(submissions_directory, only_public, dataset):
    leaderboard = _leaderboard(submissions_directory, only_public, dataset)

    def _fix_nan(value):
        if isinstance(value, float):
            if np.isnan(value):
                return None
        return value

    data = []
    for i, row in leaderboard.iterrows():
        row_data = dict(row)
        row_data = {key: _fix_nan(value) for key, value in row_data.items()} 
        row_data['config'] = _load_config(row['directory'])

        dates = get_result_dates(row['directory'])
        row_data['first_tested'] = min(dates).isoformat()
        row_data['last_tested'] = max(dates).isoformat()
        
        data.append(row_data)

    metric_names = ['IG', 'AUC', 'sAUC', 'NSS', 'CC', 'KLDiv', 'SIM'] 
    default_metric_config = {'sortInitialOrder': 'desc'}
    metrics = {metric_name: dict(default_metric_config) for metric_name in metric_names}
    metrics['KLDiv']['sortInitialOrder'] = 'asc'

    website_data = {
        'rows': data,
        'metric_names': metric_names,
        'metrics': metrics,
    }

    return website_data
    

@cli.command(context_settings={'help_option_names': ['-h', '--help']}, help="Save website data")
@click.option('--submissions-directory', '-d', default='submissions', help='read submissions in this directory')
@click.option('--only-public/--not-only-public', '-p', default=True)
@click.option('-o', '--output', default='html/data.json')
def website_data(submissions_directory, only_public, output):
    website_data = {}
    for dataset in ['MIT300', 'CAT2000']:
        website_data[dataset] = _website_data(submissions_directory, only_public, dataset)

    with open(output, 'w') as output_file:
        json.dump(website_data, output_file)


@cli.command('setup-model', context_settings={'help_option_names': ['-h', '--help']})
@click.option('--date', help="date to use as prefix for directory (default: YYYY-MM-DD)")
@click.option('--probabilistic/--no-probabilistic', '-p', help="Whether the model is probabilistic or not (default: not)")
@click.option('--location', '-l', help="location of model predictions, if available")
@click.option('-d', '--dataset', type=click.Choice(['MIT300', 'MIT1003']), default='MIT300', help='Which dataset to evaluate (default: MIT300)')
@click.option('--submissions-directory', '-d', default='submissions', help='name of submissions directory (default: "submissions")')
@click.option('--delete-source/--no-delete-source', help="Wether to delete original model predictions")
@click.argument('name')
def setup_model(date, probabilistic, location, dataset, submissions_directory, delete_source, name):
    _setup_model(date, probabilistic, location, dataset, submissions_directory, delete_source, name)


def _setup_model(date, probabilistic, location, dataset, submissions_directory, delete_source, name):
    if date is None:
        date = datetime.utcnow().strftime('%Y-%m-%d')
    model_directory = os.path.join(submissions_directory, f'{date}_{name}')

    print("Setting up submission at", model_directory)

    os.makedirs(model_directory, exist_ok=True)
    
    if location:
        if os.path.isdir(location):
            prediction_location = 'predictions'
            shutil.copytree(location, os.path.join(model_directory, prediction_location))
        elif os.path.isfile(location):
            prediction_location = os.path.basename(location)
            shutil.copy(location, os.path.join(model_directory, prediction_location))
        if delete_source:
            print("Deleting source predictions")
            if os.path.isfile(location):
                os.remove(location)
            else:
                shutil.rmtree(location)
    else:
        prediction_location = 'predictions'
        os.makedirs(os.path.join(model_directory, 'predictions'), exist_ok=True)


    config = {
        'model': {
            'filename': prediction_location,
            'name': name,
            'probabilistic': probabilistic,
        },
        'dataset': dataset,
        'public': False,
    }

    with open(os.path.join(model_directory, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        f.write("display:\n")
        f.write("  #name:\n")
        f.write("  #published:\n")
        f.write("  evaluation_comment: maps from authors\n")
        f.write("  #code:\n")

    return model_directory


@cli.command('setup-submission', context_settings={'help_option_names': ['-h', '--help']})
@click.option('--date', help="date to use as prefix for directory (default: YYYY-MM-DD)")
@click.option('--probabilistic/--no-probabilistic', '-p', help="Whether the model is probabilistic or not (default: not)")
@click.option('-d', '--dataset', type=click.Choice(['MIT300', 'MIT1003']), default='MIT300', help='Which dataset to evaluate (default: MIT300)')
@click.option('--submissions-directory', '-d', default='submissions', help='name of submissions directory (default: "submissions")')
@click.option('--delete-source/--no-delete-source', help="Wether to delete original model predictions")
@click.option('-n', '--name', help="Name of model (default: infer from submission filename)")
@click.argument('submission')
def setup_submission(date, probabilistic, dataset, submissions_directory, delete_source, name, submission):
    _setup_submission(date, probabilistic, dataset, submissions_directory, delete_source, name, submission)

def _setup_submission(date, probabilistic, dataset, submissions_directory, delete_source, name, submission):
    if name is None:
        name = os.path.splitext(os.path.basename(submission))[0]

    return _setup_model(
        date=date,
        probabilistic=probabilistic,
        location=submission,
        dataset=dataset,
        submissions_directory=submissions_directory,
        delete_source=delete_source,
        name=name)

@cli.command(help="Setup and process submission, print result email", context_settings={'help_option_names': ['-h', '--help']})
@click.option('--date', help="date to use as prefix for directory (default: YYYY-MM-DD)")
@click.option('--probabilistic/--no-probabilistic', '-p', help="Whether the model is probabilistic or not (default: not)")
@click.option('-d', '--dataset', type=click.Choice(['MIT300', 'MIT1003']), default='MIT300', help='Which dataset to evaluate (default: MIT300)')
@click.option('--submissions-directory', '-d', default='submissions', help='name of submissions directory (default: "submissions")')
@click.option('--delete-source/--no-delete-source', help="Wether to delete original model predictions", default=True)
@click.option('-n', '--name', help="Name of model (default: infer from submission filename)")
@click.argument('submission')
def process_submission(date, probabilistic, dataset, submissions_directory, delete_source, name, submission):
    location = _setup_submission(date, probabilistic, dataset, submissions_directory, delete_source, name, submission)

    execute(f'vim {location}/config.yaml')
    
    accept_results_after = '2018'
    _evaluate_location(accept_results_after, evaluation='new', location=location)
    _evaluate_location(accept_results_after, evaluation='old-python', location=location)
    _evaluate_location(accept_results_after, evaluation='old-matlab', location=location)
    
    _print_results(accept_results_after, location)



if __name__ == '__main__':
    cli()
