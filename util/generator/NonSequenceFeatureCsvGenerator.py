import numpy
import pandas
import keras.backend as K
from util.generator.CvsGeneratorBase import IGeneratorBase, GeneratorBase


class NonSequenceFeatureIGenerator(GeneratorBase):
    def __init__(self):
        super(NonSequenceFeatureIGenerator, self).__init__(0)

    def get_generator(self, data, data_root, batch_size, shuffle=False):
        return ISequenceFeatureIGenerator(data, data_root, batch_size, shuffle)

class ISequenceFeatureIGenerator(IGeneratorBase):
    def __init__(self, data, data_root, batch_size, shuffle=False, seed=None,classes=None,sample_suffix='-f.txt'):

        super(ISequenceFeatureIGenerator, self).__init__(data, data_root, batch_size, shuffle, seed, classes, sample_suffix)

        self.in_memory_data = []
        for idx, sample in enumerate(self.data):
            type, _class, filename, _ = sample
            sample_name = self.data_root + '/' + type + '/' + _class + '/' + filename + self.sample_suffix
            self.in_memory_data.append(self.get_features(sample_name).reshape(1,-1))


        self.nb_feature = len(self.in_memory_data[0][0])  # first batch sample, then first subsample

    def _get_batches_of_transformed_samples(self, index_array):

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = numpy.zeros((len(index_array),) + (self.nb_feature,), dtype=K.floatx())

        # build batch data
        for i, j in enumerate(index_array):

            batch_x[i] = numpy.array(self.in_memory_data[j])

        # build batch of labels
        batch_y = numpy.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.

        return batch_x, batch_y

    def get_features(self, path_to_sample):
        # Use a dataframe/read_csv for speed increase over numpy.
        features = pandas.read_csv(path_to_sample, sep=" ", header=None)

        return features.values