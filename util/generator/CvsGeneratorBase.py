import abc
import csv
import glob
import os

import numpy
from keras.preprocessing.image import Iterator

from thirdp.harvitronix.extract.csv_file_constats import CLASS_INDEX, NB_SUB_INDEX


class GeneratorBase():
    __metaclass__ = abc.ABCMeta

    def __init__(self, nb_seq):
        self.nb_seq = nb_seq

    def flow_from_csv_file(self, csv_file_path,batch_size, is_train):
        data = self.get_data(csv_file_path, is_train)
        data = self.filter_data(data)

        return self.flow_from_csv_data(data, os.path.dirname(csv_file_path),batch_size,is_train)

    def flow_from_csv_data(self, data, data_root,batch_size, is_train):


        return self.get_generator(data, data_root, batch_size=batch_size, shuffle=is_train)

    @abc.abstractmethod
    def get_generator(self, data, data_root, batch_size, shuffle=False):
        """ All implementors must implement this method"""
        raise Exception("not implemented")

    def get_data(self, data_file, is_train):
        """Load our data from file."""
        with open(data_file, 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        res = []
        target_type = 'Train' if is_train else 'Val'
        for sample in data:
            type, _, _, nb_sub_samples = sample
            if type == target_type:
                res.append(sample)

        return res

    def filter_data(self, samples):

        res = []
        for sample in samples:
            _, _, _, nb_sub = sample

            if int(nb_sub) > 0:
                res.append(sample)

        return res


class IGeneratorBase(Iterator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, data_root, batch_size, shuffle=False, seed=None, classes=None, sample_suffix='jpg'):

        self.data = data

        # first, count the number of samples and classes
        self.samples = len(self.data)

        if not classes:
            classes = self.get_classes(self.data)

        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        self.sample_suffix = sample_suffix

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        results = []

        sample_names = []
        self.classes = numpy.zeros((self.samples,), dtype='int32')
        self.data_root = data_root

        if self.data_root:
            for idx, sample in enumerate(self.data):
                type, _class, filename, _ = sample
                self.classes[idx] = self.class_indices[_class]
                sample_names.append(self.data_root + '/' + type + '/' + _class + '/' + filename)

        self.sample_names = numpy.array(sample_names)

        super(IGeneratorBase, self).__init__(self.samples, batch_size, shuffle, seed)

    def get_classes(self, data):
        """Extract the classes from our data."""
        classes = []
        for item in data:
            if item[CLASS_INDEX] not in classes:
                classes.append(item[CLASS_INDEX])

        # Sort them.
        classes = sorted(classes)

        # Return.
        return classes

    def get_sub_sample_paths(self, sample_path):
        """Given a path to sample (filename without extension and index), build our sample or in other name sub sample sequence.

        e.g. given C:/data/Val/angray/1
        return C:/data/Val/angray/1_001.jpg,C:/data/Val/angray/1_002.jpg,C:/data/Val/angray/1_003.jpg,..
        """
        sub_samples = sorted(glob.glob(sample_path + '*' + self.sample_suffix))

        return sub_samples

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)

            return self._get_batches_of_transformed_samples(index_array)

    @abc.abstractmethod
    def _get_batches_of_transformed_samples(self, index_array):
        """ All implementors must implement this method"""
        raise Exception("not implemented")
