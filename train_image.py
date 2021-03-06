from keras.optimizers import SGD
from work_image.abstract_train_cnn_base import TrainCnnConfiguration
from work_image.emot_train_cnn_base import EmotTrainCnnBase

# working directory
WORK_DIR = '/mnt/sda2/dev_root/work2.1/merged/image-whole/1-vggface-aligned-fer+sfew'

# specify weights file. It can be a path to a trained model or it can be one of vgg16,
# resnet or inception for imagenet weights or vggface for oxford face weights.
# WEIGHTS_FILE = 'vgg16'
# WEIGHTS_FILE = 'resnet'
# WEIGHTS_FILE = 'vggface'
WEIGHTS_FILE = '/mnt/sda2/dev_root/work2.1/merged/image-whole/1527430675.39-17403535-fer13_old_result_search_pool5_nodropout_flatten_no_seed/checkpoints/w.012-0.6754-2.91.hdf5'

# point to the dataset. Note that target dir must have data.csv
DATA_DIR = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'
BATCH_SIZE = 64
DIMENSION = 224
LR = 0.001
OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)

# img_train_gen_params={'rotation_range':20,'zoom_range':0.2,'horizontal_flip':True}
img_train_gen_params = None

config = TrainCnnConfiguration(data_dir=DATA_DIR, batch_size=BATCH_SIZE, dimension=DIMENSION, optimizer=OPTIMIZER,
                               weights_file=WEIGHTS_FILE, freeze_layer=12, reduce_lr_factor=None, reduce_lr_patience=5,
                               img_train_gen_params=img_train_gen_params, per_class_log=False, top_epochs=5)
train = EmotTrainCnnBase(work_dir=WORK_DIR, config=config)
train.train(nb_epoch=100)
