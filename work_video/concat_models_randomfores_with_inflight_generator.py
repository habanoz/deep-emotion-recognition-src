import glob
import os
import time

import keras
import numpy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.engine.topology import Input

from keras.layers.core import Dense
from keras.legacy.layers import Merge
from keras.models import load_model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn import svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from util.c_matrix import read_history_from_csv_file
from util.generator.InFlightMergeCsvGenerator import InFlightMergeGenerator
from util.generator.MergeCsvGenerator import MergeGenerator
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

    return [checkpointer,early_stopper,tensorboard,csv_logger]

def load_model_from_file(model_file):

    model = load_model(model_file)

    return model


def merge_models_sklearn_rf( nb_classes):
    model = RandomForestClassifier(n_estimators=25)
    return model

def merge_models_sklearn_lr( nb_classes):
    model = LogisticRegression(C=5.0)
    return model

def merge_models_sklearn_svc( nb_classes):
    model = svm.SVC(C=20,kernel='linear')
    return model

def main():
    model_files=['/mnt/sda2/dev_root/work2/c3d/7t-c3d__16_112_adam_b8_1lr1e6/checkpoints/w.014-0.3507-2.81.hdf5',
                 '/mnt/sda2/dev_root/work2/lstm/2_lstm_40_224/checkpoints/w.029-0.4563-1.58.hdf5']

    csv_file_path_c3d='/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_PFaces_16_112_small/data.csv'
    csv_file_path_lstm='/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Features_vggface_fc7_224/data.csv'
    seq_length_lstm=40
    seq_length_c3d=16
    VAL_BATCH_SIZE=1
    nb_classes=7
    batch_size=32
    image_dim=112

    models=[]
    for model_file in model_files:
        model = load_model(model_file)
        models.append(model)

    merged_model = merge_models_sklearn_svc(nb_classes)

    image_data_generator = ImageDataGenerator(rescale=1.0/255)
    image_generator=SequenceImageIGenerator(seq_length_c3d,image_data_generator,(image_dim,image_dim))

    feature__generator = SequenceFeatureIGenerator(seq_length_lstm)

    merged_generator_train=InFlightMergeGenerator(models,[image_generator,feature__generator],[csv_file_path_c3d,csv_file_path_lstm],True,batch_size).flow()
    merged_generator_valid=InFlightMergeGenerator(models,[image_generator,feature__generator],[csv_file_path_c3d,csv_file_path_lstm],False,VAL_BATCH_SIZE).flow()

    steps_per_epoch = len(merged_generator_train.data) / batch_size

    Xp=[]
    yp=[]
    for i in range(steps_per_epoch):
        X,y = merged_generator_train.next()
        y=numpy.argmax(y,axis=1)
        Xp.extend(X)
        yp.extend(y)
    merged_model.fit(numpy.array(Xp),numpy.array(yp))

    validation_steps = len(merged_generator_valid.data) / VAL_BATCH_SIZE

    accuracy=0.0
    for i in range(validation_steps):
        Xv,yv = merged_generator_valid.next()
        yv = numpy.argmax(yv, axis=1)
        score = merged_model.score(Xv,yv)
        print score
        accuracy=accuracy+score

    print("Final Score {}".format(accuracy/validation_steps))

    return

if __name__ == '__main__':
    main()