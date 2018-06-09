"""
Train our RNN on bottlecap or prediction files generated from our CNN.
"""
import csv
import glob
import os
import time

from thirdp.harvitronix.extract.data import DataSet
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from models import ResearchModels
from util.presentation import present_results_generator, present_results


def determine_feature_count(sequence_dir):
    sequence_files=glob.glob(sequence_dir+'/*/*/*.txt')
    first_sequence_file=sequence_files[0]

    with open(first_sequence_file, 'r') as fin:
        reader = csv.reader(fin,delimiter = ' ')
        row1=next(reader)

        return len(row1)


def train(source_dir, work_root_dir, data_type, seq_length, model, saved_model=None,
          concat=False, image_shape=None, load_to_memory=False):

    if not os.path.exists(work_root_dir):
        os.makedirs(work_root_dir)
        os.makedirs(work_root_dir+'/checkpoints')
        os.makedirs(work_root_dir+'/logs')

    # Set variables.
    nb_epoch = 2000
    batch_size = 8

    data_file=source_dir+'/data.csv'

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=work_root_dir+'/logs')

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(save_weights_only=False,
                                   filepath=work_root_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.2f}.hdf5',
                                   verbose=1,
                                   save_best_only=True, monitor='val_acc')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20, monitor='val_acc')


    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(work_root_dir+'/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    # Get the data and process it.
    if image_shape is None:
        features_length = determine_feature_count(source_dir)
        data = DataSet(
            data_file=data_file,
            sequence_dir=source_dir,
            seq_length=seq_length,
            class_limit=None
            ,given_classes=None
        )
    else:
        features_length = None
        data = DataSet(
            data_file=data_file,
            sequence_dir=source_dir,
            seq_length=seq_length,
            class_limit=None,
            image_shape=image_shape
            , given_classes=None
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    X=None
    y=None
    X_test=None
    y_test=None
    generator=None
    val_generator=None

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory(True, data_type, concat)
        X_test, y_test = data.get_all_sequences_in_memory(False, data_type, concat)

        print ("Train samples %d, test samples %d"%(len(X),len(X_test)))
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, True, data_type, concat)
        val_generator = data.frame_generator(1, False, data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model,features_length=features_length,dimension=image_shape)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        history=rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        history = rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            validation_data=val_generator,
            validation_steps=365)

    if val_generator:
        _, test=data.split_train_test()
        present_results_generator(work_root_dir, rm.model, history, val_generator, len(test),classes=data.classes)
    else:
        present_results(work_root_dir, rm.model,history, X_test=X_test, Y_test=y_test, classes=data.classes)

def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'lstm'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'conv_3d' or model == 'crnn':
        data_type = 'images'
        image_shape = (224, 224, 3)
        load_to_memory = False
    else:
        data_type = 'features'
        image_shape = None

    # MLP requires flattened features.
    if model == 'mlp':
        concat = True
    else:
        concat = False

    DATA_DIR = '/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Features_vggface_fc7_224'
    WORK_DIR = '/mnt/sda2/dev_root/work2/lstm/2_lstm_40_224'

    train(DATA_DIR, work_root_dir=WORK_DIR, data_type=data_type, model=model, concat=concat, image_shape=image_shape,
          load_to_memory=load_to_memory, seq_length=seq_length)

if __name__ == '__main__':
    main()