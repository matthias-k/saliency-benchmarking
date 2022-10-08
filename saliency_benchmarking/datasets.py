from boltons.cacheutils import cached, LRU
import pysaliency


from .constants import DATASET_LOCATION


@cached(LRU(max_size=1))
def _load_dataset(dataset_name):
    print("Loading dataset", dataset_name)
    if dataset_name.lower() == 'mit300':
        return pysaliency.get_mit300(location=DATASET_LOCATION)
    elif dataset_name.lower() == 'cat2000':
        dataset = pysaliency.get_cat2000_test(location=DATASET_LOCATION)
        dataset.cached = False
        dataset.stimuli._cache.max_size = 1
        return dataset
    elif dataset_name.lower() == 'mit1003':
        return pysaliency.get_mit1003(location=DATASET_LOCATION)
    else:
        raise ValueError("Unkown dataset", dataset_name)


def load_dataset(dataset_name):
    if dataset_name.lower() == 'mit1003':
        return _load_dataset(dataset_name.lower())[0]
    return _load_dataset(dataset_name.lower())

def get_mit300():
    return load_dataset('mit300')


def get_cat2000_test():
    return load_dataset('cat2000')


def get_mit1003_stimuli():
    return load_dataset('mit1003')


def get_mit1003():
    return _load_dataset('mit1003')
