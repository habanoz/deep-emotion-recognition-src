import csv
import os
import glob
import warnings

import numpy as np
import keras.backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, ImageDataGenerator, \
    transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis, array_to_img

from thirdp.harvitronix.extract.csv_file_constats import CLASS_INDEX

class SequenceImageGenerator(object):
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

    def __init__(self, rescale=None):
        self.rescale = rescale
        self.image_data_generator = ImageDataGenerator(rescale=rescale)
        return

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        return x

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

    def flow_from_csv(self, csv_file_path, is_train,batch_size,save_to_dir=None,target_size=(112,112),nb_seq=16):
        return CsvFileIterator(csv_file_path, is_train,batch_size=batch_size, target_size=target_size,nb_seq=nb_seq,
                               image_data_generator = self.image_data_generator, shuffle=is_train,save_to_dir=save_to_dir)


class CsvFileIterator(Iterator):
    """Iterator capable of reading images from a csv file.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, csv_file_path, is_train, image_data_generator, target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',nb_seq=16,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.csv_file_path = csv_file_path
        self.target_size = tuple(target_size)
        self.image_data_generator = image_data_generator
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical'}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical"')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.data = CsvFileIterator.get_data(csv_file_path, is_train)

        # first, count the number of samples and classes
        self.samples = len(self.data)

        if not classes:
            classes = CsvFileIterator.get_classes(self.data)

        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        results = []

        self.sample_names = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        self.data_dir = os.path.dirname(csv_file_path)
        self.nb_seq=nb_seq
        for idx, sample in enumerate(self.data):
            type, _class, filename, _ = sample
            self.classes[idx] = self.class_indices[_class]
            self.sample_names.append(self.data_dir+'/'+ type + '/' + _class + '/' + filename)

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
            if type == target_type and int(nb_sub_samples)>0:
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
        return C:/data/Val/angray/1_001.jpg,C:/data/Val/angray/1_002.jpg,C:/data/Val/angray/1_003.jpg,..
        """
        sub_samples = sorted(glob.glob(sample_path + '*jpg'))

        return sub_samples

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

            return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self, index_array):

        current_batch_size=len(index_array)

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size, self.nb_seq,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            sname = self.sample_names[j]
            sub_sample_paths = self.get_sub_sample_paths(sname)
            sample = []

            for sub_sample_path in sub_sample_paths:
                sub_sample = load_img(sub_sample_path,
                                      grayscale=grayscale,
                                      target_size=self.target_size)
                x = img_to_array(sub_sample, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                sample.append(x)
            batch_x[i] = np.array(sample)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                for seq_idx in range(len(batch_x[i])):
                    img = array_to_img(batch_x[i][seq_idx], self.data_format, scale=True)
                    fname = 'clss{_class}_{prefix}_{subindex}_{hash}.{format}'.format(
                        _class=self.classes[index_array],
                        prefix=self.save_prefix,
                        subindex=seq_idx,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.

        return batch_x, batch_y
