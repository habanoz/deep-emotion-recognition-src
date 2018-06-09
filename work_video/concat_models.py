import glob
import os
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Dense
from keras.legacy.layers import Merge
from keras.models import load_model, Sequential
from keras.optimizers import SGD

from util.generator.NonSequenceFeatureCsvGenerator import NonSequenceFeatureIGenerator
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

def get_merge_dense_only_model():
    inputs = Input(shape=(14,))
    predictions = Dense(14,)(inputs)
    predictions = Dense(7, activation='softmax')(inputs)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)

    return model

def merge_models(models, nb_classes):


    for i, model in enumerate(models):
        model.get_layer(name='flatten_1').name='flatten_1_'+str(i)
        model.get_layer(name='dense_1').name='dense_1_'+str(i)
        model.get_layer(name='dense_2').name='dense_2_'+str(i)
        model.get_layer(name='dropout_1').name='dropout_1_'+str(i)

        for layer in model.layers:
            layer.trainable=False

        model.summary()

    model=Sequential()
    model.add(Merge(models,mode='concat'))
    model.add(Dense(nb_classes,activation='softmax',trainable=True,name='dense_merge_1'))

    model.summary()

    return model

def main():
    work_dir='/mnt/sda2/dev_root/work2.1/merged/lstm_c3d/4'
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
    lr=0.0001

    if os.path.exists(work_dir):
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
            f.write('OPTIMIZER=' + 'sgd')
            f.write('\n')

    models=[]
    #for model_file in model_files:
    #    model = load_model(model_file)
    #    models.append(model)

    merged_model = get_merge_dense_only_model()

    sgd = SGD(lr=lr, momentum=0.9, decay=0, nesterov=False)
    merged_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #image_data_generator = ImageDataGenerator(rescale=1.0/255)
    #image_generator=SequenceImageIGenerator(seq_length_c3d,image_data_generator,(image_dim,image_dim))


    #feature__generator = SequenceFeatureIGenerator(seq_length_lstm)

    #merged_generator_train=MergeGenerator([image_generator,feature__generator],[csv_file_path_c3d,csv_file_path_lstm],True,batch_size)
    #merged_generator_valid=MergeGenerator([image_generator,feature__generator],[csv_file_path_c3d,csv_file_path_lstm],False,VAL_BATCH_SIZE)

    feature_generator = NonSequenceFeatureIGenerator()
    train_generator=feature_generator.flow_from_csv_file('/mnt/sda2/dev_root/dataset/combined/merged/lstmc3d/data.csv',32,True)
    val_generator=feature_generator.flow_from_csv_file('/mnt/sda2/dev_root/dataset/combined/merged/lstmc3d/data.csv',1,False)

    history = merged_model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.data) / batch_size,
        validation_data=val_generator,
        validation_steps=len(val_generator.data) / VAL_BATCH_SIZE,
        epochs=nb_epoch, callbacks=get_callbacks(work_dir), class_weight=class_weight_dict)

    list_of_model_files = glob.glob(work_dir + '/checkpoints/*.hdf5')
    latest_model_file = max(list_of_model_files, key=os.path.getctime)
    latest_model = load_model(latest_model_file)

    present_results_generator(work_dir, latest_model, history, val_generator, len(val_generator.data), classes=val_generator.class_indices)

    return

if __name__ == '__main__':
    main()