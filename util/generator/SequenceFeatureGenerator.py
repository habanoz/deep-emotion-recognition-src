import csv
import os
import glob
import pandas as pd
import numpy as np
import keras.backend as K
from keras.preprocessing.image import Iterator
from thirdp.harvitronix.extract.csv_file_constats import CLASS_INDEX


class SequenceFeatureGenerator(object):

    def __init__(self):
        return

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        raise Exception("Unsupported operation!")

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False):
        raise Exception("Unsupported operation!")

    def flow_from_csv(self, csv_file_path, is_train, batch_size, nb_seq=16):
        return CsvFileIterator(csv_file_path, is_train, batch_size=batch_size, nb_seq=nb_seq, shuffle=is_train)


class CsvFileIterator(Iterator):

    def __init__(self, csv_file_path, is_train,
                 classes=None, class_mode='categorical', nb_seq=40,
                 batch_size=8, shuffle=True, seed=None,
                 data_format=None, sample_suffix='-f.txt'):
        if data_format is None:
            data_format = K.image_data_format()
        self.csv_file_path = csv_file_path
        self.sample_suffix = sample_suffix

        self.data_format = data_format
        self.classes = classes
        if class_mode not in {'categorical'}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical"')
        self.class_mode = class_mode

        self.data = CsvFileIterator.get_data(csv_file_path, is_train)

        # first, count the number of samples and classes
        self.samples = len(self.data)

        if not classes:
            classes = CsvFileIterator.get_classes(self.data)

        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        self.sample_names = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        self.data_dir = os.path.dirname(csv_file_path)
        self.nb_seq = nb_seq

        self.in_memory_data = []
        for idx, sample in enumerate(self.data):
            type, _class, filename, _ = sample
            self.classes[idx] = self.class_indices[_class]
            sample_name = self.data_dir + '/' + type + '/' + _class + '/' + filename+self.sample_suffix
            self.sample_names.append(sample_name)
            self.in_memory_data.append(self.get_features(sample_name))

        self.nb_feature = len(self.in_memory_data[0][0]) # first batch sample, then first subsample

        super(CsvFileIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    @staticmethod
    def get_data(data_file, is_train):
        """Load our data from file."""
        with open(data_file, 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        res = []
        target_type = 'Train' if is_train else 'Val'
        for sample in data:
            type, _, _, nb_sub_samples = sample
            if type == target_type and int(nb_sub_samples) > 0:
                res.append(sample)

        return res

    @staticmethod
    def get_classes(data):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in data:
            if item[CLASS_INDEX] not in classes:
                classes.append(item[CLASS_INDEX])

        # Sort them.
        classes = sorted(classes)

        # Return.
        return classes

    def get_sub_sample_paths(self, sample_path):

        sub_samples = sorted(glob.glob(sample_path + '*jpg'))

        return sub_samples

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):


        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + (self.nb_seq, self.nb_feature), dtype=K.floatx())

        # build batch data
        for i, j in enumerate(index_array):

            batch_x[i] = np.array(self.in_memory_data[j])

        # build batch of labels
        batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.

        return batch_x, batch_y

    def get_features(self, path_to_sample):
        # Use a dataframe/read_csv for speed increase over numpy.
        features = pd.read_csv(path_to_sample, sep=" ", header=None)
        return features.values
