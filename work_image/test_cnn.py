import datetime

import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import glob
import os



from util.c_matrix import cmatrix, print_cmax, plot_confusion_matrix, cmatrix_generator

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



def test(work_dir, model_file, data_dir, dimension, prefix):
    size = (dimension, dimension)

    generators = get_generators(data_dir,size, 1)
    _, validation_generator = generators
    VALID_DATA_COUNT=len(validation_generator.classes)

    print("Loading saved model: %s." % model_file)
    model=load_model(model_file)
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




    # present results
    ts = time.time()
    timestamp=datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    results_file = work_dir + '/'+timestamp+prefix+'_results.txt'
    cm_image_file = work_dir + '/'+timestamp+prefix+'_cm.png'
    normalized_cm_image_file = work_dir + '/'+timestamp+prefix+'_cm_n.png'

    cm = cmatrix_generator(model, validation_generator,VALID_DATA_COUNT)
    validation_result = model.evaluate_generator(validation_generator, VALID_DATA_COUNT / 1)  # validation batch size = 1
    N=len(cm)

    tp = sum(cm[i][i] for i in range(N))
    fn = sum((sum( cm[i][i+1:]) for i in range(N) ))
    fp = sum( sum(cm[i][:i]) for i in range(N) )

    precision=tp*1.0/(tp+fp)
    recall=tp*1.0/(tp+fn)
    validation_result.extend([precision,recall])
    print_cmax(results_file, cm, validation_result)

    plot_confusion_matrix(cm, cm_image_file, classes=validation_generator.class_indices.keys())
    plot_confusion_matrix(cm, normalized_cm_image_file, classes=validation_generator.class_indices.keys(),normalize=True)

def main():

    WORK_DIR = '/mnt/sda2/dev_root/work2/cnn/resnet/resnet_1_224/'
    WORK_DIR = '/mnt/sda2/dev_root/work2.1/merged/image-whole/1__1512579280.12-17403535-1-vggface-aligned-catvrossloss/'
    WORK_DIR = '/mnt/sda2/dev_root/work2.1/merged/merged_models/1522678107.24-10033137-7-fer-1024-fear+sad/'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/3__1512676078.57-44395710-1-vggface-not-aligned-vggfsfew/checkpoints/w.007-0.5389-2.05.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/13__1513034264.6-10870509-vggface-alignedp01-fer+sfck/checkpoints/w.003-0.5700-1.44.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/14_1512935328.44-57039232-1-vggface-FER-NOT-ALIGNED-SFEW-CKP-YALE/checkpoints/w.010-0.6491-3.32.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/4__1512684086.44-12458347-1-vggface-aligned/checkpoints/w.013-0.6502-3.42.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/5__1512721316.9-44067573-1-vggface-aligned/checkpoints/w.004-0.5019-2.41.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/7__1512761162.37-27469082-1-vggface-aligned-fer+al-no-perturb/checkpoints/w.007-0.6593-2.86.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/2__1512672883.64-10168171-1-vggface-aligned-vggfsfew/checkpoints/w.012-0.5720-2.78.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2/cnn/resnet/resnet_1_224/checkpoints/w.025-0.59-2.72.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/1__1512579280.12-17403535-1-vggface-aligned-catvrossloss/checkpoints/w.007-0.6779-2.79.hdf5'
    MODEL_FILE = '/mnt/sda2/dev_root/work2.1/merged/merged_models/1522678107.24-10033137-7-fer-1024-fear+sad/checkpoints/w.001-0.6818-1.68.hdf5'

    DATA_DIR = '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/sfew/Aligned_PFaces_224_p01/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224_p01/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_not_aligned/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/SFEW-Processed/SFEW_PFaces_224/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/sfew/pfaces-not-aligned-224/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/CK+/pfaces-not-aligned-224/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_aligned/'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'
    DIMENSION = 224
    prefix='accuracy_precision_recall_metrics'

    prefixes=['ck_aligned','ck_not_aligned','sfew_aligned','sfew_not_aligned','fer_aligned','google_extracted_aligned','google_extracted_not_aligned']
    data=['/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224/',
          '/mnt/sda2/dev_root/dataset/CK+/pfaces-not-aligned-224/',
          '/mnt/sda2/dev_root/dataset/SFEW-Processed/SFEW_PFaces_224/',
          '/mnt/sda2/dev_root/dataset/sfew/pfaces-not-aligned-224/',
          '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/',
          '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_aligned/',
          '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_not_aligned/']

    #for pref, data_dir in zip(prefixes[5:],data[5:]):
    test(WORK_DIR, MODEL_FILE, DATA_DIR, DIMENSION,prefix)


if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
