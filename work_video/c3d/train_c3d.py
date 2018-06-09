import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.engine.training import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import model_from_json
from keras.optimizers import Adam

from util.generator.SequenceImageGenerator import SequenceImageGenerator
from util.presentation import present_results_generator


def train(work_dir, data_dir,seq_length, image_dim,batch_size, lr, optimizer,frozen_layers):
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
            f.write('DIMENSION=' + str(image_dim))
            f.write('\n')
            f.write('LR=' + str(lr))
            f.write('\n')
            f.write('OPTIMIZER=' + str(optimizer))
            f.write('\n')

    csv_file_path=data_dir+'/data.csv'

    generator=SequenceImageGenerator(rescale=1.0/255)
    train_generator=generator.flow_from_csv(csv_file_path,True,batch_size=batch_size,target_size=image_dim,nb_seq=seq_length)
    val_generator=generator.flow_from_csv(csv_file_path,False,batch_size=1,target_size=image_dim,nb_seq=seq_length)

    NB_TRAIN_SAMPLES=train_generator.samples
    NB_VAL_SAMPLES=val_generator.samples
    EPOCHS=2000
    callbacks=get_callbacks(work_dir)

    model = get_model(train_generator.num_class)

    #model.fit_generator(train_generator,NB_TRAIN_SAMPLES/batch_size,epochs=10,metrics=['accuracy'], callbacks=[],validation_data=val_generator,validation_steps=NB_VAL_SAMPLES)

    for layer in model.layers[frozen_layers:]:
        layer.trainable = True

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history=model.fit_generator(train_generator, NB_TRAIN_SAMPLES / batch_size, epochs=EPOCHS, callbacks=callbacks,
                        validation_data=val_generator, validation_steps=NB_VAL_SAMPLES)

    present_results_generator(work_dir,model,history,val_generator,NB_VAL_SAMPLES,classes=val_generator.classes)


def get_callbacks(work_dir):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath=work_dir + '/checkpoints/w.{epoch:03d}-{val_acc:.4f}-{val_loss:.2f}.hdf5',
                                   verbose=1, save_best_only=True, monitor='val_acc')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, min_lr=0.0000001)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10, monitor='val_acc')

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=work_dir + '/logs/')

    return [checkpointer,reduce_lr,early_stopper,tensorboard]


def get_model(nb_classes):
    """
    Build a 3D convolutional network, based loosely on C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    """
    # Model.

    root_dir = './thirdp/c3d_keras/'
    model_dir = root_dir + 'models'

    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    base_model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    base_model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
        print(layer.get_output_at(0).get_shape().as_list())

    x = base_model.get_layer('pool5').output

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Dropout(0.2)(x)
    x = Dense(4096)(x)
    x = Dropout(0.2)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        print(layer.get_output_at(0).get_shape().as_list())

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # model.compile(loss='mean_squared_error', optimizer='sgd')

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

    return model




def main():
    seq_length = 16
    image_dim = (112, 112)
    batch_size=8

    DATA_DIR = '/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_PFaces_16_112/'
    WORK_DIR = '/mnt/sda2/dev_root/work2/c3d/7t-c3d__16_112_adam_b8_1lr1e6'
    LR = 1e-5
    OPTIMIZER = Adam(lr=LR, decay=1e-4)
    frozen_layers = 0

    train(WORK_DIR, DATA_DIR, seq_length, image_dim,batch_size,LR,optimizer=OPTIMIZER,frozen_layers=frozen_layers)

if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
