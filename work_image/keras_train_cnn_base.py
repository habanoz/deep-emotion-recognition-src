import glob
import os
import time

import numpy
from keras import backend as B
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LambdaCallback
from keras.engine.training import Model
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from util.c_matrix import cmatrix_generator
from work_image.VggFaceE3 import VGGFace
from work_image.abstract_train_cnn_base import AbstractTrainCnnBase, SHUFFLE, SEED, VAL_BATCH_SIZE, EARLY_STOP_PATIENCE

LOSS = 'categorical_crossentropy'


class KerasTrainCnnBase(AbstractTrainCnnBase):
    def __init__(self, work_dir, config, nb_classes):

        super(KerasTrainCnnBase, self).__init__(work_dir, config, nb_classes)
        return

    def load_model_from_file(self, weights_file):
        return load_model(weights_file)

    def get_best_trained_model_weights(self):
        list_of_model_files = glob.glob(self.work_dir + '/checkpoints/*.hdf5')
        best_model_file = max(list_of_model_files, key=os.path.getctime)
        return best_model_file

    def get_generators(self, save_images=False):
        if self.config.img_train_gen_params:
            train_datagen = ImageDataGenerator(rescale=1. / 255, **self.config.img_train_gen_params)
        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

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

    def get_callbacks(self,base_only=False):
        timestamp = time.time()
        csv_logger = CSVLogger(self.work_dir + '/logs/training-' + str(timestamp) + '.log')

        res = [csv_logger]

        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath=self.work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.2f}.hdf5',
            verbose=1, save_best_only=True, monitor='val_acc')

        res.append(checkpointer)

        def on_epoch_end_callback(epoch, logs):

            train_log = []
            val_log = []

            if self.config.per_class_log:
                train_gen, val_gen = self.generators

                confusion_matrix_train = cmatrix_generator(self.model, train_gen, train_gen.samples)
                for i, row in enumerate(confusion_matrix_train):
                    train_log.append(row[i] * 1.0 / numpy.sum(row))

                confusion_matrix_val = cmatrix_generator(self.model, val_gen, val_gen.samples)
                for i, row in enumerate(confusion_matrix_val):
                    val_log.append(row[i] * 1.0 / numpy.sum(row))

                logs['train_per_class'] = train_log
                logs['val_per_class'] = val_log

                # keep a copy
                self.__last_logs = logs

        lambda_callback = LambdaCallback(on_epoch_end=on_epoch_end_callback)

        res.append(lambda_callback)

        if base_only:
            return res

        # Helper: Stop training when we stop learning.
        early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

        res.append(early_stopper)

        if self.config.reduce_lr_factor and self.config.reduce_lr_patience:
            reduce_lr = ReduceLROnPlateau(factor=self.config.reduce_lr_factor, patience=self.config.reduce_lr_patience,
                                          min_lr=0.0000001, verbose=1)
            res.append(reduce_lr)

        return res

    def get_optimizer_string(self):
        o = ""
        if isinstance(self.config.optimizer, str):
            o=o+('OPTIMIZER=' + str(self.config.optimizer) + str('\n'))
        else:
            o = o +('OPTIMIZER class=' + str(type(self.config.optimizer).__name__) + '\n')
            if hasattr(self.config.optimizer, 'lr'):
                o = o +('OPTIMIZER lr=' + str(float(B.get_value(self.config.optimizer.lr))) + str('\n'))
            if hasattr(self.config.optimizer, 'decay'):
                o = o +('OPTIMIZER decay=' + str(float(B.get_value(self.config.optimizer.decay))) + str('\n'))
            if hasattr(self.config.optimizer, 'momentum'):
                o = o +('OPTIMIZER momentum=' + str(float(B.get_value(self.config.optimizer.momentum))) + str('\n'))
        return o

    def get_train_val_acc(self):
        train_generator, val_generators = self.generators
        train_result = self.model.evaluate_generator(train_generator, self.TRAIN_DATA_COUNT / self.config.batch_size)
        val_result = self.model.evaluate_generator(val_generators, self.VALID_DATA_COUNT)
        return train_result, val_result

    def train_model(self, nb_epoch=100, callbacks=None, adjust_class_weights=False):
        train_generator, validation_generator = self.generators

        class_weight_dict = None
        if adjust_class_weights:
            labels = train_generator.classes
            class_weight = compute_class_weight('balanced', numpy.unique(labels), labels)
            class_weight_dict = dict(enumerate(class_weight))

        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=self.TRAIN_DATA_COUNT / self.config.batch_size,
            epochs=nb_epoch,
            callbacks=callbacks, class_weight=class_weight_dict,
            validation_data=validation_generator,
            validation_steps=self.VALID_DATA_COUNT / VAL_BATCH_SIZE,
            shuffle=SHUFFLE
        )

        return history.history

    def load_model_for_training(self):

        if not self.model:
            self.train_top = True
            if self.config.weights_file == "vggface":
                base_model = VGGFace(include_top=False,input_shape=(self.config.dimension,self.config.dimension,3))
                self.model = self.get_model_with_classification_head(base_model)
            elif self.config.weights_file == "vgg16":
                base_model = VGG16(include_top=False,input_shape=(self.config.dimension,self.config.dimension,3))
                self.model = self.get_model_with_classification_head(base_model)
            elif self.config.weights_file == "inception":
                base_model = InceptionV3(include_top=False,input_shape=(self.config.dimension,self.config.dimension,3))
                self.model = self.get_model_with_classification_head(base_model)
            elif self.config.weights_file == "resnet":
                base_model = ResNet50(include_top=False,input_shape=(self.config.dimension,self.config.dimension,3))
                self.model = self.get_model_with_classification_head(base_model)
            else:
                self.model = self.load_model_from_file(self.config.weights_file)
                self.train_top = False
                self.model.summary()

    def get_model_with_classification_head(self, base_model):

        base_model.summary()

        x = base_model.output
        x = GlobalAveragePooling2D(name='flatten_1')(x)
        #x = Flatten(name='flatten_1')(x)
        x = Dense(1024, activation='relu', name='fc6_1')(x)
        x = Dense(1024, activation='relu', name='fc7_2')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.nb_classes, activation='softmax', name='fc8_3')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer="rmsprop", loss=LOSS, metrics=['accuracy'])
        model.summary()

        return model

    def prepare_model_for_training(self):
        # set first layers frozen
        for layer in self.model.layers[:self.config.freeze_layers]:
            layer.trainable = False
        for layer in self.model.layers[self.config.freeze_layers:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        self.model.compile(optimizer=self.config.optimizer, loss=LOSS, metrics=['accuracy'])

