import sys

from keras.optimizers import SGD
from work_image.abstract_train_cnn_base import TrainCnnConfiguration, change_seed
from work_image.merged_train_cnn import MergedModelCnn


# stack base models
# - resnet
# - vgg16
# - simplified vggface
#
# suitable for running multiple times using run_train_image_merge_ptest_sample.sh

if len(sys.argv) <= 1:
    raise Exception("provide work dir argument")

WORK_DIR = sys.argv[1]
DATA_DIR = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'
BATCH_SIZE = 16
DIMENSION = 224
LR = 0.001

OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)
img_train_gen_params = None

config = TrainCnnConfiguration(data_dir=DATA_DIR, batch_size=BATCH_SIZE, dimension=DIMENSION, optimizer=OPTIMIZER,
                               weights_file=None, freeze_layer=12, reduce_lr_factor=None, reduce_lr_patience=5,
                               img_train_gen_params=img_train_gen_params, top_epochs=8)

change_seed(None)

model_urls = [
    '/mnt/sda2/dev_root/deep-emotion-recognition/models/1522605364.22-12593475-resnet-fer13/checkpoints/w.014-0.6121-2.29.hdf5',
    '/mnt/sda2/dev_root/deep-emotion-recognition/models/1522396968.84-16963257-vgg16-fer13/checkpoints/w.032-0.6464-1.15.hdf5',
    '/mnt/sda2/dev_root/work2.1/merged/image-whole/1__1512579280.12-17403535-1-vggface-aligned-catvrossloss/sample_ttest_noseed/1526928528.82-17403535-sample_7/checkpoints/w.017-0.6949-1.72.hdf5']

merge_extractors = []
train = MergedModelCnn(work_dir=WORK_DIR, config=config, model_urls=model_urls)
train.train(nb_epoch=0)
