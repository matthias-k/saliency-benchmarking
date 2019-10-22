import os


def iterate_submissions(root_directory):
    """Iterate over all submissions in nested directory.
    
    Submissions are detected by the presence of a config.yaml.
    Directories starting with an underscore or a dot are ignored.
    """
    if os.path.isfile(os.path.join(root_directory, 'config.yaml')):
        yield root_directory

    for dir_entry in sorted(os.scandir(root_directory), key=lambda dir_entry: dir_entry.name):
        if dir_entry.name.startswith('_'):
            continue
        if dir_entry.name.startswith('.'):
            continue
        if dir_entry.is_dir():
            yield from iterate_submissions(dir_entry.path)
