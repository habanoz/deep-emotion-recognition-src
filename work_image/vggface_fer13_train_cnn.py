from util.generator.KerasImageGenerator import ImageDataGenerator
from work_image.abstract_train_cnn_base import SHUFFLE,SEED
from work_image.emot_train_cnn_base import EmotTrainCnnBase


class CustomGeneratorsTrain(EmotTrainCnnBase):

    def __init__(self, work_dir, config):
        EmotTrainCnnBase.__init__(self, work_dir,config=config)

    def train(self,nb_epoch=100):
        EmotTrainCnnBase.train(self,nb_epoch)

    def get_generators(self, save_images=False):
        """
        This implementation uses custom generators instead of original keras generators

        :param save_images:
        :return:
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        save_to_dir=None
        if save_images:
            save_to_dir=self.work_dir+'/saved'

        train_generator = train_datagen.flow_from_directory(
            self.config.data_dir + 'Train/',
            target_size=self.config.size,
            batch_size=self.config.batch_size,
            save_to_dir=save_to_dir,
            classes=None,
            class_mode='categorical',shuffle=SHUFFLE,seed=SEED)


        # batch_size=1 and shuffle=False; we want validation data be exactly same,
        # validation accuracy exactly same for the same model and same data
        validation_generator = test_datagen.flow_from_directory(
            self.config.data_dir + 'Val/',
            target_size=self.config.size,
            batch_size=1,
            classes=None,
            class_mode='categorical', shuffle=False)

        return train_generator, validation_generator
