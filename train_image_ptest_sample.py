
import sys

from keras.optimizers import SGD
from work_image.abstract_train_cnn_base import TrainCnnConfiguration, AbstractTrainCnnBase
from work_image.emot_train_cnn_base import EmotTrainCnnBase

# train a model.
# suitable for running muliple times using run_train_image_ptest_sample.sh
##############################################################################

if len(sys.argv)<=1:
    raise Exception("provide work dir argument")

WORK_DIR = sys.argv[1]

WEIGHTS_FILE = 'vggface'
DATA_DIR = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'

BATCH_SIZE = 64
DIMENSION = 224
LR = 0.001
OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)

AbstractTrainCnnBase.VALID_DATA_COUNT=2
img_train_gen_params={'rotation_range':20,'zoom_range':0.2,'horizontal_flip':True}
img_train_gen_params=None

config=TrainCnnConfiguration(data_dir=DATA_DIR, batch_size=BATCH_SIZE, dimension=DIMENSION, optimizer=OPTIMIZER,weights_file=WEIGHTS_FILE,freeze_layer=12,
                             reduce_lr_factor=None,reduce_lr_patience=5,img_train_gen_params=img_train_gen_params, per_class_log=False,top_epochs=5 )
train = EmotTrainCnnBase(work_dir=WORK_DIR, config=config)
train.train(nb_epoch=100)