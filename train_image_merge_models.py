from keras.optimizers import SGD
from work_image.abstract_train_cnn_base import TrainCnnConfiguration, change_seed
from work_image.merged_train_cnn import MergedModelCnn

# stack exppert models
# - fear expert
# - sad expert
# - simplified vggface
#
# more experts can be added to training by commenting them out.
# keep in mind, more models may require more memory.


WORK_DIR = '/mnt/sda2/dev_root/work2.1/merged/merged_models/7-fer-1024x512-0_001-fear+sad'
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
    # '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514448648.57-17280232-Angry/checkpoints/w.015-0.7661-1.01.hdf5',
    '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514450283.12-16129103-Fear/checkpoints/w.012-0.7229-1.84.hdf5',
    '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514451621.75-11419305-Sad/checkpoints/w.010-0.7495-1.04.hdf5',
    # '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514453226.77-74565569-Surprise/checkpoints/w.009-0.8868-0.94.hdf5',
    # '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514454327.81-11283002-Happy/checkpoints/w.012-0.9038-0.83.hdf5',
    # '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514459192.19-58728062-Neutral/checkpoints/w.005-0.7833-0.74.hdf5',
    # '/mnt/sda2/dev_root/work2.1/merged/oneVrest/4_balanced_val/1514460934.35-17929388-Disgust/checkpoints/w.009-0.8294-1.18.hdf5',
    '/mnt/sda2/dev_root/work2.1/merged/image-whole/1__1512579280.12-17403535-1-vggface-aligned-catvrossloss/checkpoints/w.007-0.6779-2.79.hdf5']

merge_extractors = []
train = MergedModelCnn(work_dir=WORK_DIR, config=config, model_urls=model_urls)
train.train(nb_epoch=0)
