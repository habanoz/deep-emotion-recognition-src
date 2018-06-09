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
import time
import numpy
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger, LambdaCallback
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects
from keras_vggface.vggface2 import VGGFace
from sklearn.utils.class_weight import compute_class_weight

from util.presentation import present_results_generator
from numpy.random import seed
from tensorflow import set_random_seed
import random

FREEZE_LAYERS = None
VAL_BATCH_SIZE = 1
NB_CLASSES = 7

SHUFFLE=True
SEED=1

if SHUFFLE and SEED:
    seed(1)
    set_random_seed(2)
    random.seed(1)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 2)


get_custom_objects().update({'top_2_categorical_accuracy':top_2_accuracy})


def get_generators(data_dir,size, batch_size,work_dir=None):
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)


    if work_dir:
        save_to_dir=work_dir+'/saved'

    save_to_dir = None

    train_generator = train_datagen.flow_from_directory(
        data_dir + 'Train/',
        target_size=size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        classes=None,
        class_mode='categorical',shuffle=SHUFFLE,seed=SEED)


    # batch_size=1 and shuffle=False; we want validation data be exactly same,
    # validation accuracy exactly same for the same model and same data
    validation_generator = test_datagen.flow_from_directory(
        data_dir + 'Val/',
        target_size=size,
        batch_size=1,
        classes=None,
        class_mode='categorical', shuffle=False)

    return train_generator, validation_generator


def get_vgg16_face_model(input_shape,weights='imagenet'):
    hidden_dim = 1024
    global FREEZE_LAYERS

    FREEZE_LAYERS = 4

    vgg_model = VGGFace(include_top=False, input_shape=input_shape)

    # first: train only the top layers (which were randomly initialized)
    for layer in vgg_model.layers:
        layer.trainable = False

    print("VGG16-Face has {} layers".format(len(vgg_model.layers)))

    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(NB_CLASSES, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)

    return custom_vgg_model


def get_vgg16_model(weights='imagenet'):
    global FREEZE_LAYERS
    FREEZE_LAYERS = 4

    # create the base pre-trained model
    base_model = VGG16(weights=weights, include_top=False)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    print("VGG16 has {} layers".format(len(base_model.layers)))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(NB_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def get_resnet_model(weights='imagenet'):
    global FREEZE_LAYERS
    FREEZE_LAYERS = 100

    # create the base pre-trained model
    base_model = ResNet50(weights=weights, include_top=False)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    print("Resnet has {} layers".format(len(base_model.layers)))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(NB_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def get_inception_model(weights='imagenet'):
    # Inception has 311 layers
    global FREEZE_LAYERS
    FREEZE_LAYERS = 172

    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    print("Inception has {} layers".format(len(base_model.layers)))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(NB_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def get_top_layer_model(model):
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_mid_layer_model(model, optimizer):
    """After we fine-tune the dense layers, train deeper."""
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def get_callbacks(work_dir):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath=work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.2f}.hdf5',
                                   verbose=1, save_best_only=True, monitor='val_acc')

    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000001,verbose=1)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)

    # Helper: TensorBoard
    #tensorboard = TensorBoard(log_dir=work_dir + '/logs/')

    timestamp = time.time()
    csv_logger = CSVLogger(work_dir+'/logs/training-' + \
        str(timestamp) + '.log')

    def lambda_callback(batch, logs) :
        print(batch)

    batch_print_callback = LambdaCallback(on_batch_begin=lambda_callback)

    return [checkpointer,early_stopper,csv_logger,reduce_lr,batch_print_callback]

def train_model(model, nb_epoch, generators,batch_size,train_data_count,validation_data_count,callbacks=[]):
    train_generator, validation_generator = generators

    labels = train_generator.classes
    class_weight = compute_class_weight('balanced', numpy.unique(labels), labels)
    class_weight_dict = dict(enumerate(class_weight))
    class_weight_dict=None

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_data_count / batch_size,
        epochs=nb_epoch,
        callbacks=callbacks, class_weight=class_weight_dict,
        validation_data=validation_generator,
        validation_steps=validation_data_count / VAL_BATCH_SIZE,
        shuffle=SHUFFLE
    )

    return model, history


def train(model,work_dir,weights_file,data_dir,batch_size,dimension,lr,optimizer):
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
        os.makedirs(work_dir + '/saved')
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
            f.write('LR='+str(lr))
            f.write('\n')
            f.write('OPTIMIZER='+str(optimizer))
            f.write('\n')


    generators = get_generators(data_dir,size, batch_size,work_dir)

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = get_top_layer_model(model)
        model, _ = train_model(model, 10, generators,batch_size,TRAIN_DATA_COUNT,VALID_DATA_COUNT)

        model.save(work_dir + '/checkpoints/m.hdf5')
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = get_mid_layer_model(model,optimizer)
    model, history = train_model(model, 10, generators, batch_size,TRAIN_DATA_COUNT,VALID_DATA_COUNT, get_callbacks(work_dir))

    _, validation_generator = generators


    present_results_generator(work_dir, model, history, validation_generator, len(validation_generator.classes),
                              classes=validation_generator.class_indices, suffix='last')

    list_of_model_files = glob.glob(work_dir + '/checkpoints/*.hdf5')
    latest_model_file = max(list_of_model_files, key=os.path.getctime)
    latest_model = load_model(latest_model_file)

    present_results_generator(work_dir, latest_model, None, validation_generator, len(validation_generator.classes),
                              classes=validation_generator.class_indices,suffix='best')


def main():
    WORK_DIR = '/mnt/sda2/dev_root/work2/cnn/vggface/vggface_16_224_fe13_then_sfew/'
    WEIGHTS_FILE = '/mnt/sda2/dev_root/work2/cnn/vggface/vggface_2_224/checkpoints/w.012-0.6682-3.03.hdf5'
    WEIGHTS_FILE = None
    DATA_DIR = '/mnt/sda2/dev_root/dataset/SFEW-Processed/SFEW_PFaces_224/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/FER13-Processed/FER13-224/'
    BATCH_SIZE = 64
    DIMENSION = 224
    LR = 1e-6
    #OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)
    OPTIMIZER = Adam(lr=LR, decay=1e-6)

    LR = 0.001
    OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)

    train(get_vgg16_face_model(input_shape=(DIMENSION,DIMENSION,3)), WORK_DIR,WEIGHTS_FILE,DATA_DIR,BATCH_SIZE,DIMENSION,LR,OPTIMIZER)


if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
