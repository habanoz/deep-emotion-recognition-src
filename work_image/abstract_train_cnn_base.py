import glob
import os
import random
from abc import ABCMeta, abstractmethod
from time import time
from numpy.random import seed
from tensorflow import set_random_seed
from util.presentation import present_results_generator

PARAMS_FILE_NAME = "params.txt"
FREEZE_LAYERS = None
VAL_BATCH_SIZE = 1
SHUFFLE = True
SEED = None
EARLY_STOP_PATIENCE = 15


def change_seed(new_seed):
    global SEED
    SEED = new_seed


if SEED:
    print("SEED is " + str(SEED))
    seed(SEED)
    set_random_seed(SEED)
    random.seed(SEED)
else:
    print("SEED is None")


def generate_directories(work_dir, timestamp, hash):
    """ Generate directory structure which is required to save results
        If directory exists and contains files, work dir is modifed by appending timestamp

        returns work_dir which is different than input if directory already exists
    """

    list_of_saved_model_files = glob.glob(work_dir + '/checkpoints/*')

    # if directory already exists and contains files rename it using timestamp
    work_dir_basename = os.path.basename(os.path.normpath(work_dir))
    work_dir_dirname = os.path.dirname(os.path.normpath(work_dir))
    work_dir = work_dir_dirname + '/' + str(timestamp) + '-' + str(hash)[0:8] + '-' + work_dir_basename

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.makedirs(work_dir + '/logs')
        os.makedirs(work_dir + '/checkpoints')

    return work_dir


class TrainCnnConfiguration():
    def __init__(self, weights_file, data_dir, batch_size, dimension, optimizer, freeze_layer=None, per_class_log=False,
                 reduce_lr_factor=None, reduce_lr_patience=None,
                 img_train_gen_params=None, top_epochs=10):
        self.weights_file = weights_file
        self.data_dir = data_dir
        self.dimension = dimension
        self.size = (dimension, dimension)
        self.input_shape = (dimension, dimension, 3)
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.freeze_layers = freeze_layer
        self.per_class_log = per_class_log

        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.img_train_gen_params = img_train_gen_params
        self.top_epochs = top_epochs

        return

    def to_string(self):
        o = ""
        o = o + ('WEIGHTS_FILE=' + str(self.weights_file) + str('\n'))
        o = o + ('DATA_DIR=' + str(self.data_dir) + str('\n'))
        o = o + ('BATCH_SIZE=' + str(self.batch_size) + str('\n'))
        o = o + ('DIMENSION=' + str(self.dimension) + str('\n'))
        o = o + ('per_class_log=' + str(self.per_class_log) + str('\n'))
        o = o + ('freeze_layers=' + str(self.freeze_layers) + str('\n'))
        o = o + ('per_class_log=' + str(self.per_class_log) + str('\n'))
        o = o + ('reduce_lr_factor=' + str(self.reduce_lr_factor) + str('\n'))
        o = o + ('reduce_lr_patience=' + str(self.reduce_lr_patience) + str('\n'))
        o = o + ('seed=' + str(SEED) + str('\n'))

        return o

    def to_hash(self):
        return hash(self.to_string())


def sorted_keys_by_values(class_indices):
    """
    take {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3} and return ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    :param class_indices:
    :return:
    """
    sorted_keys = ['NAN'] * len(class_indices)
    for key, value in class_indices.iteritems():
        sorted_keys[int(value)] = key

    return sorted_keys


class AbstractTrainCnnBase:
    __metaclass__ = ABCMeta

    def __init__(self, work_dir, config, nb_classes):

        self.config = config
        self.nb_classes = nb_classes

        self.TRAIN_DATA_COUNT = sum([len(files) for r, d, files in os.walk(self.config.data_dir + 'Train')])
        self.VALID_DATA_COUNT = sum([len(files) for r, d, files in os.walk(self.config.data_dir + 'Val')])

        self.model = None

        self.generators = self.get_generators()
        self.class_indices = sorted_keys_by_values(self.generators[0].class_indices)
        hash_val = hash(self.get_optimizer_string()) + self.config.to_hash()

        # initialize
        self.timestamp = time()
        self.__last_logs = None
        self.train_top = False

        self.work_dir = generate_directories(work_dir, self.timestamp, hash_val)
        print("Work Dir {}".format(os.path.realpath(self.work_dir)))
        self.report_params()

        return

    def report_params(self):
        # save configuration parameters
        with open(self.work_dir + "/" + PARAMS_FILE_NAME, 'w') as f:
            f.write(self.config.to_string())
            f.write('\n')
            f.write('TRAIN_DATA_COUNT=' + str(self.TRAIN_DATA_COUNT))
            f.write('\n')
            f.write('VALID_DATA_COUNT=' + str(self.VALID_DATA_COUNT))
            f.write('\n')
            f.write("***\n")
            f.write(self.get_optimizer_string())

    @abstractmethod
    def train_model(self, nb_epoch=100, callbacks=None, adjust_class_weights=False):
        raise NotImplementedError

    @abstractmethod
    def get_train_val_acc(self):
        raise NotImplementedError

    def train(self, nb_epoch=100):
        self.load_model_for_training()

        train_result, val_result = None, None  # self.get_train_val_acc()

        top_logs = None
        if self.train_top:
            top_logs = self.train_model(nb_epoch=self.config.top_epochs, callbacks=self.get_callbacks(base_only=True))

        self.prepare_model_for_training()

        logs = self.train_model(nb_epoch=nb_epoch, callbacks=self.get_callbacks())

        if top_logs:
            for key in top_logs:
                logs['top_' + key] = top_logs[key]

        if val_result:
            logs['init_train_loss'] = train_result[0]
            logs['init_train_acc'] = train_result[1]
            logs['init_val_loss'] = val_result[0]
            logs['init_val_acc'] = val_result[1]

        self.present_results(logs)

    def present_results(self, logs):
        """
        Plot accuracy and loss figures
        Do this for last model which is result of all training epochs. This model is probably overtrained.

        Do this for best model which is the one with highest success metric.

        :param logs:
        :return:
        """
        _, validation_generator = self.generators
        present_results_generator(self.work_dir, self.model, logs, validation_generator,
                                  self.VALID_DATA_COUNT, classes=self.class_indices,
                                  suffix='last', train_top_epochs=(self.config.top_epochs if self.train_top else None))

        best_model_weights = self.get_best_trained_model_weights()
        best_model = self.load_model_from_file(best_model_weights)

        present_results_generator(self.work_dir, best_model, None, validation_generator,
                                  self.VALID_DATA_COUNT, classes=self.class_indices,
                                  suffix='best')

    @abstractmethod
    def prepare_model_for_training(self):
        """Before starting model training, prepare the model e.g. compile parameters"""
        raise NotImplementedError

    @abstractmethod
    def get_generators(self):
        """Return (train_generator, val_generator) """
        raise NotImplementedError

    @abstractmethod
    def get_callbacks(self, base_only=False):
        """ List of callbacks for training procedure"""
        raise NotImplementedError

    @abstractmethod
    def load_model_from_file(self, weights_file):
        """Given weights file, load model from disk
        File must contain model metadata together with weights.
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_trained_model_weights(self):
        """ After training is over, find and return the model file with highest success metric"""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer_string(self):
        raise NotImplementedError

    @abstractmethod
    def load_model_for_training(self):
        raise NotImplementedError


def main():
    pass


if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
