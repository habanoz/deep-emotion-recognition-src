import datetime
import os
import time

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from util.c_matrix import print_cmax, plot_confusion_matrix, cmatrix_generator

FREEZE_LAYERS = None
VAL_BATCH_SIZE = 1
NB_CLASSES = 7


def get_generators(data_dir, size, batch_size):
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


def test(work_dir, model_file, data_dir, dimension, prefix):
    size = (dimension, dimension)

    generators = get_generators(data_dir, size, 1)
    _, validation_generator = generators
    VALID_DATA_COUNT = len(validation_generator.classes)

    print("Loading saved model: %s." % model_file)
    model = load_model(model_file)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # present results
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    results_file = work_dir + '/' + timestamp + prefix + '_results.txt'
    cm_image_file = work_dir + '/' + timestamp + prefix + '_cm.png'
    normalized_cm_image_file = work_dir + '/' + timestamp + prefix + '_cm_n.png'

    cm = cmatrix_generator(model, validation_generator, VALID_DATA_COUNT)
    validation_result = model.evaluate_generator(validation_generator,
                                                 VALID_DATA_COUNT / 1)  # validation batch size = 1
    N = len(cm)

    tp = sum(cm[i][i] for i in range(N))
    fn = sum((sum(cm[i][i + 1:]) for i in range(N)))
    fp = sum(sum(cm[i][:i]) for i in range(N))

    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    validation_result.extend([precision, recall])
    print_cmax(results_file, cm, validation_result)

    plot_confusion_matrix(cm, cm_image_file, classes=validation_generator.class_indices.keys())
    plot_confusion_matrix(cm, normalized_cm_image_file, classes=validation_generator.class_indices.keys(),
                          normalize=True)


def main():
    WORK_DIR = '/mnt/sda2/dev_root/work2.1/merged/merged_models/1522678107.24-10033137-7-fer-1024-fear+sad/'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/merged_models/1522678107.24-10033137-7-fer-1024-fear+sad/checkpoints/w.001-0.6818-1.68.hdf5'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'
    DIMENSION = 224
    prefix = 'accuracy_precision_recall_metrics'

    prefixes = ['ck_aligned', 'ck_not_aligned', 'sfew_aligned', 'sfew_not_aligned', 'fer_aligned',
                'google_extracted_aligned', 'google_extracted_not_aligned']
    data = ['/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224/',
            '/mnt/sda2/dev_root/dataset/CK+/pfaces-not-aligned-224/',
            '/mnt/sda2/dev_root/dataset/SFEW-Processed/SFEW_PFaces_224/',
            '/mnt/sda2/dev_root/dataset/sfew/pfaces-not-aligned-224/',
            '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/',
            '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_aligned/',
            '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_not_aligned/']

    # for pref, data_dir in zip(prefixes[5:],data[5:]):
    test(WORK_DIR, MODEL_FILE, DATA_DIR, DIMENSION, prefix)


if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
