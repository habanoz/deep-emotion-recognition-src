from keras.optimizers import SGD
from work_image.abstract_train_cnn_base import TrainCnnConfiguration
from work_image.keras_train_cnn_base import KerasTrainCnnBase

# train one vs all models (a.k.a expert models) for each emotion.
#

for emot in ['Fear', 'Sad']:
    WORK_DIR = '/mnt/sda2/dev_root/work2.1/merged/oneVrest/emotiw_oneVrest_vggface6949_averagepool/' + emot
    WEIGHTS_FILE = 'vggface'
    DATA_DIR = '/mnt/sda2/dev_root/dataset/combined/fer2013_224_NEW_CLEAN-ONE-VS-REST-SUBSAMPLED-VAl-BALANCED/' + emot + '_VS_rest/'

    BATCH_SIZE = 64
    DIMENSION = 224
    LR = 0.001
    OPTIMIZER = SGD(lr=LR, momentum=0.9, decay=0.0005)

    img_train_gen_params = None

    config = TrainCnnConfiguration(data_dir=DATA_DIR, batch_size=BATCH_SIZE, dimension=DIMENSION, optimizer=OPTIMIZER,
                                   weights_file=WEIGHTS_FILE, freeze_layer=12, reduce_lr_factor=None,
                                   reduce_lr_patience=5, img_train_gen_params=img_train_gen_params)
    train = KerasTrainCnnBase(work_dir=WORK_DIR, config=config, nb_classes=2)
    train.train(nb_epoch=0)
