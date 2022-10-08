from abc import ABC, abstractmethod
import glob
import errno
import os
from random import shuffle
from time import sleep
import pickle
from datetime import datetime

from boltons.fileutils import atomic_save
from boltons.iterutils import chunked

import numpy as np


class DistributedSaver(ABC):
    """Manages a shared datastore"""
    def __init__(self, filename, initial_data):
        self.filename = filename
        self.initial_data = initial_data

        self.data = self._read_data(self.filename)

    @abstractmethod
    def read_data(self, filehandle):
        pass

    @abstractmethod
    def write_data(self, filehandle, data):
        pass

    @abstractmethod
    def merge_data(self, old_data, new_data):
        pass

    @abstractmethod
    def update_indices(self, old_data, indices, updates):
        pass

    def _read_data(self, cache_file):
        # have a separate file that is only overwritten
        # to indicate that we already have a cache
        # even it is not there right now due to the
        # atomic save
        if os.path.isfile(cache_file+'.exists') or os.path.isfile(cache_file):
            while not os.path.isfile(cache_file):
                sleep(0.001)
            while True:
                try:
                    with open(cache_file, 'rb') as f:
                        #print("Loading previous data")
                        data = self.read_data(f)
                except OSError as e:
                    if e.errno == errno.ESTALE:
                        print("stale file handle, waiting")
                        sleep(5 + np.random.rand()*10)
                        continue
                    else:
                        raise e
                break
        else:
            data = self.initial_data
        return data

    def _save_data(self, cache_file, data):
        with atomic_save(cache_file) as f:
            self.write_data(f, data)
        with open(cache_file+'.exists', 'w') as f:
            f.write('exists')

        # create a backup every hour in case we loose data via a race condition
        backup_filename = cache_file + datetime.utcnow().strftime('.%Y-%m-%d_%H')
        if not os.path.isfile(backup_filename):
            print("Creating backup", backup_filename)
            with open(backup_filename, 'wb') as f:
                self.write_data(f, data)

    def update_data(self, indices, updates):
        old_data = self.data
        return self._update_data(old_data, self.filename, indices, updates)

    def _update_data(self, old_data, cache_file, indices, updates):
        while True:
            try:
                data = self._update_data_inner(old_data, cache_file, indices, updates)
            except (FileExistsError, OSError) as e:
                # FileExistsError: somebody else is writing right now
                # OSError: most likely stale file handle due to overwritten file
                # in both cases: try again
                print(e)
                sleep(0.1)
                if np.random.rand() > 0.9999:
                    # with a small probability, delete the part file in case it's from a killed process
                    print("Deleting leftover part file")
                    part_file = cache_file + '.part'
                    os.remove(part_file)
            else:
                break
        return data

    def _update_data_inner(self, old_data, cache_file, indices, updates):
        data = self._read_data(cache_file)
        data = self.merge_data(old_data, data)
        data = self.update_indices(data, indices, updates)
        self._save_data(cache_file, data)
        self.data = data
        return data

    def get_missing_indices(self):
        return [k for k, item in enumerate(self.data) if item is None]

    def __len__(self):
        return len(self.data)

    @property
    def done_count(self):
        return len(self) - len(self.get_missing_indices())

    def get_batch(self, batch_size, placeholder=None, random=True, random_start=True):
        """Get a batch of entries that are not yet computed."""
        missing_indices = self.get_missing_indices()
        this_size = min(batch_size, len(missing_indices))

        if random:
            return np.random.choice(missing_indices, size=this_size, replace=False)
        elif random_start:
            print("Random start")
            indices = np.zeros(len(self), dtype=bool)
            indices[missing_indices] = True

            batch_indices = list(range(int(np.ceil(len(self) / batch_size))))
            shuffle(batch_indices)
            for batch_index in batch_indices:
                print('test batch')
                batch_start = batch_index * batch_size
                batch_end = min(batch_start + batch_size, len(self))
                batch = indices[batch_start:batch_end]
                if np.any(batch):
                    print('found match')
                    return batch_start + np.nonzero(batch)[0]



            #print("get batches")
            #batches = list(chunked(range(len(self)), batch_size))
            #shuffle(batches)
            #for batch in batches:
            #    print("test batch")
            #    real_batch = [index for index in batch if index in missing_indices]
            ##    print("Found match")
            #    if real_batch:
            #        return real_batch
            #start = np.random.randint(0, len(missing_indices))
            #return missing_indices[start:start + batch_size]
        else:
            return missing_indices[:batch_size]

    def cleanup(self):
        """Remove backup files"""
        backup_pattern = self.filename + '.20*'
        for backup_file in glob.glob(backup_pattern):
            print("Removing {backup_file}".format(backup_file=backup_file))
            os.remove(backup_file)
        exists_file = self.filename + '.exists'
        if os.path.exists(exists_file):
            print("Removing {exists_file}".format(exists_file=exists_file))
            os.remove(exists_file)


class DistributedPickleSaver(DistributedSaver):
    """Saves pickle files that contains dict of lists"""
    def read_data(self, filehandle):
        return pickle.load(filehandle)

    def write_data(self, filehandle, data):
        pickle.dump(data, filehandle)

    def merge_data(self, old_data, new_data):
        for key in old_data.keys():
            for i, value in enumerate(new_data[key]):
                if value is None and old_data[key][i] is not None:
                    # there seems to be a race condition which sometimes leads to
                    # a worker using empty data instead of loading the old data
                    # (probably when `isfile` executed exactly when the atomic save takes place)
                    print("Some data went missing, replacing with old values!")
                    new_data[key][i] = old_data[key][i]
        return new_data

    def update_indices(self, old_data, indices, updates):
        for k, index in enumerate(indices):
            for key, value in updates.items():
                old_data[key][index] = value[k]
        return old_data

    def __len__(self):
        test_key = list(self.data.keys())[0]
        return len(self.data[test_key])

    def get_missing_indices(self):
        test_key = list(self.data.keys())[0]
        return [k for k, item in enumerate(self.data[test_key]) if item is None]


class DistributedNPZSaver(DistributedSaver):
    """Saves npz files that contains dict of arrays"""
    def read_data(self, filehandle):
        with np.load(filehandle) as npz_file:
            data = dict(npz_file)
        return data

    def write_data(self, filehandle, data):
        np.savez(filehandle, **data)

    def merge_data(self, old_data, new_data):
        for key in old_data.keys():
            for i, value in enumerate(new_data[key]):
                if value is None and old_data[key][i] is not None:
                    # there seems to be a race condition which sometimes leads to
                    # a worker using empty data instead of loading the old data
                    # (probably when `isfile` executed exactly when the atomic save takes place)
                    print("Some data went missing, replacing with old values!")
                    new_data[key][i] = old_data[key][i]
        return new_data

    def update_indices(self, old_data, indices, updates):
        for k, index in enumerate(indices):
            #print(index)
            for key, value in updates.items():
                #assert not np.isnan(value[k])
                if np.isnan(value[k]):
                    print("WARNING: NaN value found at index", index)
                old_data[key][index] = value[k]
        return old_data

    def __len__(self):
        test_key = list(self.data.keys())[0]
        return len(self.data[test_key])

    def get_missing_indices(self):
        test_key = list(self.data.keys())[0]
        #return [k for k, item in enumerate(self.data[test_key]) if np.all(np.isnan(item))]
        is_missing_per_key = []
        for key in self.data:
            key_data = self.data[key]
            key_missing = np.isnan(key_data)
            if key_data.ndim > 1:
                key_missing = np.all(key_missing, axis=0)
            is_missing_per_key.append(key_missing)

        #is_missing_per_key = [[np.all(np.isnan(item)) for item in self.data[key]] for key in self.data]
        is_missing = np.any(np.vstack(is_missing_per_key), axis=0)
        missing_inds = np.arange(len(is_missing))[is_missing]
        return list(missing_inds)
        #return [k for k, item in enumerate(is_missing) if item]


class DistributedNPYSaver(DistributedSaver):
    """Saves pickle files that a npy array"""
    def read_data(self, filehandle):
        return np.load(filehandle)

    def write_data(self, filehandle, data):
        np.save(filehandle, data)

    def merge_data(self, old_data, new_data):
        missing_inds = (~np.isnan(old_data)) & np.isnan(new_data)
        if missing_inds.sum():
            print("Some data went missing, replacing with old values:", missing_inds.sum())
        new_data[missing_inds] = old_data[missing_inds]
        return new_data

    def update_indices(self, old_data, indices, updates):
        for i, value in zip(indices, updates):
            old_data[i] = value
        return old_data

    def get_missing_indices(self):
        return [k for k, item in enumerate(self.data) if np.all(np.isnan(item))]
