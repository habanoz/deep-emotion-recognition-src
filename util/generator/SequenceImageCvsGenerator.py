import keras.backend as K
import numpy
from keras.preprocessing.image import load_img, img_to_array
from util.generator.CvsGeneratorBase import IGeneratorBase, GeneratorBase


class SequenceImageIGenerator(GeneratorBase):
    def __init__(self, nb_seq,image_data_generator,target_size):
        self.image_data_generator=image_data_generator
        self.target_size=target_size

        super(SequenceImageIGenerator, self).__init__(nb_seq)

    def get_generator(self, data, data_root, batch_size, shuffle=False):
        return ISequenceImageIGenerator(data,data_root,self.image_data_generator,self.target_size,batch_size,self.nb_seq,shuffle)


class ISequenceImageIGenerator(IGeneratorBase):
    def __init__(self, data, data_dir,image_data_generator,target_size, batch_size, nb_seq,
                 shuffle=False, seed=None,classes=None,sample_suffix='jpg',data_format=None,color_mode='rgb',class_mode='categorical'):

        self.image_data_generator=image_data_generator
        self.target_size=target_size
        self.nb_seq=nb_seq

        if data_format is None:
            data_format = K.image_data_format()

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
            raise ValueError('Invalid class_mode:', class_mode,'; expected one of "categorical"')

        self.class_mode = class_mode

        super(ISequenceImageIGenerator, self).__init__(data, data_dir, batch_size, shuffle=shuffle, seed=seed, classes=classes, sample_suffix=sample_suffix)


    def _get_batches_of_transformed_samples(self, index_array):

        current_batch_size=len(index_array)

        #print "ISequenceImageIGenerator"
        #print (self.sample_names[index_array])

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = numpy.zeros((current_batch_size, self.nb_seq,) + self.image_shape, dtype=K.floatx())
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
            batch_x[i] = numpy.array(sample)

        # build batch of labels
        batch_y = numpy.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.

        return batch_x, batch_y