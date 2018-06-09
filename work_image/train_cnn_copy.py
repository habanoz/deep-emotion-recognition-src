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
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects
from keras_vggface.vggface2 import VGGFace
from sklearn.utils.class_weight import compute_class_weight

from util.c_matrix import print_cmax, plot_confusion_matrix, cmatrix_generator


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 2)


get_custom_objects().update({'top_2_categorical_accuracy':top_2_accuracy})

FREEZE_LAYERS = None
VAL_BATCH_SIZE = 1
NB_CLASSES = 7

def get_generators(data_dir,size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        data_dir + 'Train/',
        target_size=size,
        batch_size=batch_size,
        classes=None,
        class_mode='categorical')

    # batch_size=1 and shuffle=False; we want validation data be exactly same,
    # validation accuracy exactly same for the same model and same data
    validation_generator = test_datagen.flow_from_directory(
        data_dir + 'Val/',
        target_size=size,
        batch_size=1,
        classes=None,
        class_mode='categorical', shuffle=False)

    return train_generator, validation_generator


def get_vgg16_face_model(input_shape=(224,224,3),weights='imagenet'):
    hidden_dim_1 = 1024
    hidden_dim_2 = 1024
    global FREEZE_LAYERS

    FREEZE_LAYERS = 9

    vgg_model = VGGFace(include_top=False, input_shape=input_shape)

    # first: train only the top layers (which were randomly initialized)
    for layer in vgg_model.layers:
        layer.trainable = False

    print("VGG16-Face has {} layers".format(len(vgg_model.layers)))

    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim_1, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim_2, activation='relu', name='fc7')(x)
    out = Dense(NB_CLASSES, activation='softmax', name='fc8')(x)

    custom_vgg_model = Model(vgg_model.input, out)

    custom_vgg_model.summary()

    print(custom_vgg_model.get_config())

    return custom_vgg_model


def get_vgg16_model(weights='imagenet'):
    global FREEZE_LAYERS
    FREEZE_LAYERS = 13

    # create the base pre-trained model
    base_model = VGG16(weights=weights, include_top=False)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    print("VGG16 has {} layers".format(len(base_model.layers)))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
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
    if NB_CLASSES==2:
        loss='binary_crossentropy'
    else:
        loss='categorical_crossentropy'


    model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])

    return model


def get_mid_layer_model(model, optimizer):
    """After we fine-tune the dense layers, train deeper."""
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    if NB_CLASSES==2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'


    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy','top_2_categorical_accuracy'])
        #,weighted_metrics=['accuracy','top_2_categorical_accuracy'])

    return model


def get_callbacks(work_dir):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath=work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.4f}.hdf5',verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=11)

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=work_dir + '/logs/')

    return [checkpointer,reduce_lr,early_stopper,tensorboard]

def train_model(model, nb_epoch, generators,batch_size,train_data_count,validation_data_count,callbacks=[]):
    train_generator, validation_generator = generators

    labels=train_generator.classes
    class_weight = compute_class_weight('balanced', numpy.unique(labels), labels)
    class_weight_dict = dict(enumerate(class_weight))

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_data_count / batch_size,
        validation_data=validation_generator,
        validation_steps=validation_data_count / VAL_BATCH_SIZE,
        epochs=nb_epoch,
        callbacks=callbacks,class_weight=None)

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


    generators = get_generators(data_dir,size, batch_size)

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = get_top_layer_model(model)
        model, _ = train_model(model, 3, generators,batch_size,TRAIN_DATA_COUNT,VALID_DATA_COUNT)

        model.save(work_dir + '/checkpoints/m.hdf5')
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file,by_name=True)

    # Get and train the mid layers.
    model = get_mid_layer_model(model,optimizer)
    model, history = train_model(model, 100, generators, batch_size,TRAIN_DATA_COUNT,VALID_DATA_COUNT, get_callbacks(work_dir))

    _, validation_generator = generators
    list_of_model_files = glob.glob(work_dir + '/checkpoints/*.hdf5')
    latest_model_file = max(list_of_model_files, key=os.path.getctime)
    latest_model = load_model(latest_model_file)

    # present results
    results_file = work_dir + '/results.txt'
    cm_image_file = work_dir + '/cm.png'
    normalized_cm_image_file = work_dir + '/cm_n.png'
    history_image_file = work_dir + '/history.png'

    confusion_matrix = cmatrix_generator(latest_model, validation_generator, VALID_DATA_COUNT,nb_classes=NB_CLASSES)
    validation_result = model.evaluate_generator(validation_generator,
                                                 VALID_DATA_COUNT / 1)  # validation batch size = 1
    #print_cmax(results_file, confusion_matrix, validation_result)
    #plot_histor(history, history_image_file)
    #plot_confusion_matrix(confusion_matrix, cm_image_file, classes=validation_generator.class_indices.keys())
    #plot_confusion_matrix(confusion_matrix, normalized_cm_image_file, classes=validation_generator.class_indices.keys(),normalize=True)

def main():
    WORK_DIR = '/mnt/sda2/dev_root/work2/cnn/vggface/vggface_14_224_fe13_sfew_yale_ckp/'
    WEIGHTS_FILE = '/mnt/sda2/dev_root/work2/cnn/vggface/vggface_2_224/checkpoints/w.012-0.6682-3.03.hdf5'
    WEIGHTS_FILE = None
    DATA_DIR = '/mnt/sda2/dev_root/dataset/combined/FER13-SFEW-YALE-CKP-224/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/FER13-Processed/FER13-224/'
    BATCH_SIZE = 64
    DIMENSION = 224
    LR = 1e-6
    #OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)
    OPTIMIZER = Adam(lr=LR, decay=1e-6),

    train(get_vgg16_face_model(input_shape=(DIMENSION,DIMENSION,3)), WORK_DIR,WEIGHTS_FILE,DATA_DIR,BATCH_SIZE,DIMENSION,LR,OPTIMIZER)


if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
