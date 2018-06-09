import numpy
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from scikitlearn import KerasClassifier
from sklearn_booster import AdaBoostClassifier
import os

# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)

# Function to create model, required for KerasClassifier
def create_model():
    WEIGHTS_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/1__1512579280.12-17403535-1-vggface-aligned-catvrossloss/checkpoints/w.007-0.6779-2.79.hdf5'
    model = load_model(WEIGHTS_FILE)

    for layer in model.layers[0:12]:
        layer.trainable=False
    for layer in model.layers[12:]:
        layer.trainable = True

    LR = 0.001
    OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)

    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

def get_data(data_dir,size,batch_size):
    """
    This implementation uses custom generators instead of original keras generators

    :param save_images:
    :return:
    """
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_dir=data_dir + '/Train/'
    val_dir=data_dir + '/Val/'

    TRAIN_DATA_COUNT = sum([len(files) for r, d, files in os.walk(train_dir)])
    VALID_DATA_COUNT = sum([len(files) for r, d, files in os.walk(val_dir)])



    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=size,
        batch_size=TRAIN_DATA_COUNT,
        classes=None,
        class_mode='categorical',shuffle=True,seed=SEED)


    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=size,
        batch_size=VALID_DATA_COUNT,
        classes=None,
        class_mode='categorical', shuffle=False)

    train_data=train_generator.next()
    val_data=validation_generator.next()

    return train_data, val_data

data_dir='/mnt/sda2/dev_root/dataset/combined/balanced-unaligned-14-9-48x48'

train,val=get_data(data_dir=data_dir,size=(224,224),batch_size=None)
X,y=train

# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=32, verbose=0)
classifier = AdaBoostClassifier(base_estimator=model,n_estimators=10,learning_rate=0.001,algorithm='SAMME.R')
classifier.fit(X,y)
