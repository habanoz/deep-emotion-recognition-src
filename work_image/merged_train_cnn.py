from keras.engine.training import Model
from keras.layers.core import Dense
from keras.models import Sequential, load_model

from util.generator.KerasMultiModelImageGenerator import ImageDataGenerator
from work_image.abstract_train_cnn_base import SHUFFLE,SEED
from work_image.emot_train_cnn_base import EmotTrainCnnBase


class MergedModelCnn(EmotTrainCnnBase):

    def __init__(self, work_dir, config, model_urls):
        self.merge_extractors = []
        self.input_size=0
        for model_file in model_urls:
            _model = load_model(model_file)
            new_model = Model(
                inputs=_model.input,
                outputs=_model.layers[-2].output
            )
            # assumes output is in form of (None,Dimension)
            self.input_size+=new_model.output.shape[1].value

            self.merge_extractors.append(new_model)

        EmotTrainCnnBase.__init__(self, work_dir,config=config)

    def train(self,nb_epoch=100):
        EmotTrainCnnBase.train(self,nb_epoch)

    def load_model_for_training(self):
        model = Sequential()
        model.add(Dense(1024, input_shape=(3072,)))
        model.add(Dense(512))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.compile(optimizer="sgd",
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model=model
        self.train_top=True

    def get_generators(self, save_images=False):
        """
        This implementation uses custom generators instead of original keras generators

        :param save_images:
        :return:
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255,merge_extractors=self.merge_extractors)

        test_datagen = ImageDataGenerator(rescale=1. / 255,merge_extractors=self.merge_extractors)

        save_to_dir = None
        if save_images:
            save_to_dir = self.work_dir + '/saved'

        train_generator = train_datagen.flow_from_directory(
            self.config.data_dir + 'Train/',
            target_size=self.config.size,
            batch_size=self.config.batch_size,
            save_to_dir=save_to_dir,
            classes=None,
            class_mode='categorical', shuffle=SHUFFLE, seed=SEED)

        # batch_size=1 and shuffle=False; we want validation data be exactly same,
        # validation accuracy exactly same for the same model and same data
        validation_generator = test_datagen.flow_from_directory(
            self.config.data_dir + 'Val/',
            target_size=self.config.size,
            batch_size=1,
            classes=None,
            class_mode='categorical', shuffle=False)

        return train_generator, validation_generator

