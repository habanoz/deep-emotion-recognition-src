import glob
import os
import time

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.engine.topology import Input
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from util.generator.InFlightMergeCsvGenerator import InFlightMergeGenerator
from util.generator.SequenceFeatureCsvGenerator import SequenceFeatureIGenerator
from util.generator.SequenceImageCvsGenerator import SequenceImageIGenerator
from util.presentation import present_results_generator


def get_callbacks(work_dir):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.3f}-{val_loss:.3f}.hdf5', save_best_only=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)

    timestamp = time.time()
    csv_logger = CSVLogger(work_dir + '/logs/' + 'training-' + str(timestamp) + '.log')

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=work_dir + '/logs/')

    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, min_lr=0.0000001)

    return [checkpointer,early_stopper,tensorboard,csv_logger]

def load_model_from_file(model_file):

    model = load_model(model_file)

    return model


def merge_models( nb_classes):

    input=Input(shape=(14,))
    #dense1 = keras.layers.Dense(nb_classes*2,name='dense_merge_2')(input)
    #dropout1 = keras.layers.Dropout(0.3,name='dense_dropout_1')(dense1)
    out = keras.layers.Dense(nb_classes,activation='softmax',name='dense_merge_1')(input)
    model = keras.models.Model(inputs=input, outputs=out)

    model.summary()

    return model

def main():
    work_dir='/mnt/sda2/dev_root/work2.1/merged/lstm_c3d/13-merge-data-top-only-model-single-layer-adagrad'
    model_files=['/mnt/sda2/dev_root/work2/c3d/7t-c3d__16_112_adam_b8_1lr1e6/checkpoints/w.014-0.3507-2.81.hdf5',
                 '/mnt/sda2/dev_root/work2/lstm/2_lstm_40_224/checkpoints/w.029-0.4563-1.58.hdf5']

    csv_file_path_c3d='/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_PFaces_16_112_small/data.csv'
    csv_file_path_lstm='/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Features_vggface_fc7_224/data.csv'
    class_weight_dict=None
    seq_length_lstm=40
    seq_length_c3d=16
    VAL_BATCH_SIZE=1
    nb_epoch=1000
    nb_classes=7
    batch_size=32
    image_dim=112
    lr=0.001
    optimizer=SGD(lr=lr, momentum=0.9, decay=0.0005)
    optimizer='adagrad'

    if os.path.exists(work_dir):
       print("Work dir exists. Choose another!")
       raise Exception("Work dir exists. Choose another!")
    else:
        os.makedirs(work_dir)
        os.makedirs(work_dir + '/logs')
        os.makedirs(work_dir + '/checkpoints')
        # save configuration parameters
        with open(work_dir + "/params.txt", 'w') as f:
            f.write('\n')
            f.write('BATCH_SIZE=' + str(batch_size))
            f.write('\n')
            f.write('DIMENSION=' + str(image_dim))
            f.write('\n')
            f.write('LR=' + str(lr))
            f.write('\n')
            f.write('OPTIMIZER=' + str(optimizer))
            f.write('\n')

    models=[]
    for model_file in model_files:
        model = load_model(model_file)
        models.append(model)

    merged_model = merge_models(nb_classes)

    merged_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    with open(work_dir + "/params.txt", 'a') as f:
        merged_model.summary(print_fn=lambda x: f.write(x + '\n'))

    image_data_generator = ImageDataGenerator(rescale=1.0/255)
    image_generator=SequenceImageIGenerator(seq_length_c3d,image_data_generator,(image_dim,image_dim))

    feature__generator = SequenceFeatureIGenerator(seq_length_lstm)

    merged_generator_train=InFlightMergeGenerator(models,[image_generator,feature__generator],[csv_file_path_c3d,csv_file_path_lstm],True,batch_size).flow()
    merged_generator_valid=InFlightMergeGenerator(models,[image_generator,feature__generator],[csv_file_path_c3d,csv_file_path_lstm],False,VAL_BATCH_SIZE).flow()

    image_generator_valid = merged_generator_valid.igenerators[0]
    feature_generator_valid = merged_generator_valid.igenerators[1]

    c3d_model = models[0]
    lstm_model = models[1]

    #present_results_generator(work_dir, c3d_model, None, image_generator_valid, len(merged_generator_valid.data),classes=merged_generator_valid.class_indices, suffix='c3d')
    #present_results_generator(work_dir, lstm_model, None, feature_generator_valid, len(merged_generator_valid.data),classes=merged_generator_valid.class_indices, suffix='lstm')


    history = merged_model.fit_generator(
       merged_generator_train,
       steps_per_epoch=len(merged_generator_train.data) / batch_size,
       validation_data=merged_generator_valid, validation_steps=len(merged_generator_valid.data)  / VAL_BATCH_SIZE,
        epochs=nb_epoch, callbacks=get_callbacks(work_dir), class_weight=class_weight_dict)

    present_results_generator(work_dir, merged_model, history, merged_generator_valid, len(merged_generator_valid.data),classes=merged_generator_valid.class_indices,suffix='last')

    present_results_generator(work_dir, c3d_model, None, image_generator_valid, len(merged_generator_valid.data), classes=merged_generator_valid.class_indices, suffix='c3d-post')
    present_results_generator(work_dir, lstm_model, None, feature_generator_valid, len(merged_generator_valid.data), classes=merged_generator_valid.class_indices, suffix='lstm-post')

    list_of_model_files = glob.glob(work_dir + '/checkpoints/*.hdf5')
    latest_model_file = max(list_of_model_files, key=os.path.getctime)
    latest_model = load_model(latest_model_file)

    present_results_generator(work_dir, latest_model, None, merged_generator_valid, len(merged_generator_valid.data), classes=merged_generator_valid.class_indices)

    return

if __name__ == '__main__':
    main()