import csv
import os
import glob
import pandas as pd
import numpy as np
import keras.backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img

from extract.exctract_landmarks_fix_seq_length import SQ_LM_FILE_SUFFIX
from thirdp.harvitronix.extract.csv_file_constats import CLASS_INDEX


class SequenceLandMarkGenerator(object):
    """Generate minibatches of image sequnces data (e.g. video sample) with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

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
    # TODO: change descriptions
    """Iterator capable of reading images from a csv file.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.

        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"categorical"`: categorical targets,
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling
    """

    def __init__(self, csv_file_path, is_train,
                 classes=None, class_mode='categorical', nb_seq=40,
                 batch_size=8, shuffle=True, seed=None,
                 data_format=None, sample_suffix=None):
        if data_format is None:
            data_format = K.image_data_format()
        self.csv_file_path = csv_file_path

        self.sample_suffix=None
        if sample_suffix:
            self.sample_suffix = sample_suffix
        else:
            self.sample_suffix=SQ_LM_FILE_SUFFIX

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
            type, _class, filename, nb_subsample = sample
            if int(nb_subsample)<self.nb_seq:
                continue

            self.classes[idx] = self.class_indices[_class]
            sample_name = self.data_dir + '/' + type + '/' + _class + '/' + filename+self.sample_suffix
            self.sample_names.append(sample_name)
            self.in_memory_data.append(self.get_landmarks(sample_name))

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
        """Given a path to sample (filename without extension and index), build our sample or in other name sub sample sequence.

        e.g. given C:/data/Val/angray/1
        return C:/data/Val/angry/1_001.jpg,C:/data/Val/angry/1_002.jpg,C:/data/Val/angry/1_003.jpg,..
        """
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

    def get_landmarks(self, path_to_sample):
        return np.load(path_to_sample)
