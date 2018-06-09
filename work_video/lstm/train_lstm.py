import os
import time

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from util.generator.SequenceFeatureGenerator import SequenceFeatureGenerator
from util.presentation import present_results_generator


def train(work_dir, data_dir,seq_length,batch_size, lr, optimizer):
    if os.path.exists(work_dir):
       raise Exception("Work dir exists. Choose another!")
    else:
        os.makedirs(work_dir)
        os.makedirs(work_dir + '/logs')
        os.makedirs(work_dir + '/checkpoints')
        # save configuration parameters
        with open(work_dir + "/params.txt", 'w') as f:
            f.write('\n')
            f.write('DATA_DIR=' + str(data_dir))
            f.write('\n')
            f.write('BATCH_SIZE=' + str(batch_size))
            f.write('\n')
            f.write('SEQ_LENGTH=' + str(seq_length))
            f.write('\n')
            f.write('LR=' + str(lr))
            f.write('\n')
            f.write('OPTIMIZER=' + str(optimizer))
            f.write('\n')

    csv_file_path=data_dir+'/data.csv'

    generator=SequenceFeatureGenerator()
    train_generator=generator.flow_from_csv(csv_file_path,True,batch_size=batch_size,nb_seq=seq_length)
    val_generator=generator.flow_from_csv(csv_file_path,False,batch_size=1,nb_seq=seq_length)

    NB_TRAIN_SAMPLES=train_generator.samples
    NB_VAL_SAMPLES=val_generator.samples
    EPOCHS=2000
    callbacks=get_callbacks(work_dir)

    model = get_model(train_generator.num_class,(seq_length,train_generator.nb_feature))

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history=model.fit_generator(train_generator, NB_TRAIN_SAMPLES / batch_size, epochs=EPOCHS, callbacks=callbacks,
                        validation_data=val_generator, validation_steps=NB_VAL_SAMPLES)

    present_results_generator(work_dir,model,history,val_generator,NB_VAL_SAMPLES,classes=val_generator.class_indices.keys())


def get_callbacks(work_dir):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath=work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.2f}.hdf5',
                                   verbose=1, save_best_only=True, monitor='val_acc')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, min_lr=0.0000001)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10, monitor='val_acc')

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=work_dir + '/logs/')

    timestamp = time.time()
    csv_logger = CSVLogger(work_dir+'/logs/training-' + \
        str(timestamp) + '.log')

    return [checkpointer,reduce_lr,early_stopper,tensorboard,csv_logger]

def get_model(nb_classes, input_shape):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def main():
    seq_length = 40
    DATA_DIR = '/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Features_vggface_fc7_224'
    WORK_DIR = '/mnt/sda2/dev_root/work2/lstm/4xxxx'
    LR = 1e-4
    batch_size=8
    OPTIMIZER = Adam(lr=LR, decay=1e-4)
    train(WORK_DIR, DATA_DIR, seq_length=seq_length,batch_size=batch_size,lr=LR,optimizer=OPTIMIZER)

if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
