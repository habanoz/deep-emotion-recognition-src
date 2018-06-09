"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
import glob
import os

import numpy
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers.core import Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from sklearn.utils.class_weight import compute_class_weight

from util.c_matrix import print_cmax, plot_confusion_matrix, plot_history, cmatrix_generator
from util.generator.SequenceLandMarkGenerator import SequenceLandMarkGenerator

VAL_BATCH_SIZE = 1
NB_CLASSES = 7

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 2)


get_custom_objects().update({'top_2_categorical_accuracy':top_2_accuracy})


def get_generators(data_dir, nb_seq, batch_size):
    train_datagen = SequenceLandMarkGenerator()

    test_datagen = SequenceLandMarkGenerator()

    train_generator = train_datagen.flow_from_csv( data_dir + '/data.csv',True, batch_size=batch_size,nb_seq=nb_seq)

    # batch_size=1 and shuffle=False; we want validation data be exactly same,
    # validation accuracy exactly same for the same model and same data
    validation_generator = test_datagen.flow_from_csv( data_dir + '/data.csv',False, batch_size=batch_size,nb_seq=nb_seq)

    return train_generator, validation_generator



def lstm_model(input_shape):

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape,dropout=0.5))
    model.add(LSTM(128, return_sequences=True,dropout=0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES, activation='softmax'))

    return model


def get_mid_layer_model(model, optimizer):
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']) #,'top_2_categorical_accuracy'],weighted_metrics=['accuracy','top_2_categorical_accuracy'])

    return model


def get_callbacks(work_dir):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath=work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.2f}.hdf5',
                                   verbose=1, save_best_only=True, monitor='val_acc')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20, monitor='val_acc')

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=work_dir + '/logs/')

    return [checkpointer,early_stopper,tensorboard]

def train_model(model, nb_epoch, generators,batch_size,train_data_count,validation_data_count,callbacks=[]):
    train_generator, validation_generator = generators

    labels = train_generator.classes
    class_weight = compute_class_weight('balanced', numpy.unique(labels), labels)
    class_weight_dict = dict(enumerate(class_weight))
    class_weight_dict=None

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_data_count / batch_size,
        validation_data=validation_generator,
        validation_steps=validation_data_count / VAL_BATCH_SIZE,
        epochs=nb_epoch,
        callbacks=callbacks,class_weight=class_weight_dict)

    return model, history


def train(model,work_dir,weights_file,data_dir,batch_size,dimension,lr,optimizer,nb_seq):

    # OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)
    optimizer = Adam(lr=lr, decay=1e-6)

    size = (dimension, dimension)
    INPUT_SHAPE = (dimension, dimension, 3)
    TRAIN_DATA_COUNT = sum([len(files) for r, d, files in os.walk(data_dir + 'Train')])
    VALID_DATA_COUNT = sum([len(files) for r, d, files in os.walk(data_dir + 'Val')])


    if os.path.exists(work_dir):
        raise Exception("Work dir exists. Choose another!")
    else:
        os.makedirs(work_dir)
        os.makedirs(work_dir + '/logs')
        os.makedirs(work_dir + '/checkpoints')
        # save configuration parameters
        with open(work_dir+"/params.txt",'w') as f:
            f.write('WEIGHTS_FILE=='+str(weights_file))
            f.write('\n')
            f.write('DATA_DIR='+str(data_dir))
            f.write('\n')
            f.write('BATCH_SIZE='+str(batch_size))
            f.write('\n')
            f.write('DIMENSION='+str(dimension))
            f.write('\n')
            f.write('TRAIN_DATA_COUNT='+str(TRAIN_DATA_COUNT))
            f.write('\n')
            f.write('VALID_DATA_COUNT='+str(VALID_DATA_COUNT))
            f.write('\n')
            f.write('nb_seq='+str(nb_seq))
            f.write('\n')
            f.write('LR='+str(lr))
            f.write('\n')
            f.write('OPTIMIZER='+str(optimizer))
            f.write('\n')


    generators = get_generators(data_dir,nb_seq, batch_size)


    if weights_file:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = get_mid_layer_model(model,optimizer)
    model, history = train_model(model, 1000, generators, batch_size,TRAIN_DATA_COUNT,VALID_DATA_COUNT, get_callbacks(work_dir))

    with open(work_dir + "/params.txt", 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    _, validation_generator = generators
    list_of_model_files = glob.glob(work_dir + '/checkpoints/*.hdf5')
    latest_model_file = max(list_of_model_files, key=os.path.getctime)
    latest_model = load_model(latest_model_file)

    # present results
    results_file = work_dir + '/results.txt'
    cm_image_file = work_dir + '/cm.png'
    normalized_cm_image_file = work_dir + '/cm_n.png'
    history_image_file = work_dir + '/history.png'

    confusion_matrix = cmatrix_generator(latest_model, validation_generator, VALID_DATA_COUNT)
    validation_result = model.evaluate_generator(validation_generator,
                                                 VALID_DATA_COUNT / 1)  # validation batch size = 1
    print_cmax(results_file, confusion_matrix, validation_result)
    plot_history(history, history_image_file)
    plot_confusion_matrix(confusion_matrix, cm_image_file, classes=validation_generator.class_indices.keys())
    plot_confusion_matrix(confusion_matrix, normalized_cm_image_file, classes=validation_generator.class_indices.keys(),
                          normalize=True)

def main():
    WORK_DIR = '/mnt/sda2/dev_root/work2/lm-lstm/lstm_3_16/'
    WEIGHTS_FILE = None
    DATA_DIR = '/mnt/sda2/dev_root/dataset/CK+/cohn-kanade-lm-224-71fx/'
    BATCH_SIZE = 64
    DIMENSION = 224
    NB_LANDMARKS=23
    LR = 1e-6
    #OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)
    OPTIMIZER = Adam(lr=LR, decay=1e-6),
    NB_SEQ = 16

    train(lstm_model(input_shape=(NB_SEQ,NB_LANDMARKS)), WORK_DIR,WEIGHTS_FILE,DATA_DIR,BATCH_SIZE,DIMENSION,LR,OPTIMIZER,NB_SEQ)


if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
