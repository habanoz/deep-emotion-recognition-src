import csv
import os
import numpy
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import glob
import itertools
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cmatrix_generator(model, generator, nb_files, nb_classes=7):
    cmat = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    generated = 0
    while generated < nb_files:
        #Xlist, ylist = generator.next()
        Xlist, ylist = generator.next()
        for X, y in zip([Xlist], [ylist]):
            #prediction = model.predict(np.array([X]))
            prediction = model.predict(X)
            prediction_idx = np.argmax(prediction,axis=1)
            actual_idx = np.argmax(y,axis=1)
            for i,j in zip(actual_idx,prediction_idx):
                cmat[i, j] = 1 + cmat[i, j]
                generated = generated + 1
    return cmat


def cmatrix(model, Xlist, ylist, nb_classes=7):
    cmat = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    generated = 0
    for X, y in zip(Xlist, ylist):
        prediction = model.predict(np.array([X]))
        prediction_idx = np.argmax(prediction)
        actual_idx = np.argmax(y)
        cmat[actual_idx, prediction_idx] = 1 + cmat[actual_idx, prediction_idx]
        generated = generated + 1

    return cmat




def generate_cmax(data_dir, model, img_dims, batch_size, out_dir):
    valid_generator = ImageDataGenerator(rescale=1. / 255)
    valid_files = glob.glob(data_dir + '/Val/*/*.*')

    validation_generator = valid_generator.flow_from_directory(
        data_dir + 'Val/',
        target_size=img_dims,
        batch_size=batch_size,
        classes=None,
        class_mode='categorical',
        shuffle=False)

    cmat = cmatrix(model, validation_generator, len(valid_files))
    result = model.evaluate_generator(validation_generator, len(valid_files) / batch_size)

    print_cmax(out_dir + '/cm.txt', cmat, result)
    plot_confusion_matrix(cmat,out_dir+'/cm.png',validation_generator.class_indices.values(),normalize=True)

    return

def plot_confusion_matrix(cm, file, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file)
    plt.close()
    plt.clf()

def plot_acc_history(logs, file,vertical_line=None):
    plt.figure()

    train_accuracy=[]
    val_accuracy=[]

    if 'acc' in logs:
        train_accuracy=logs['acc']

    if 'val_acc' in logs:
        val_accuracy = logs['val_acc']

    if 'top_acc' in logs and 'top_val_acc' in logs and vertical_line:
        train_accuracy=logs['top_acc']+train_accuracy
        val_accuracy = logs['top_val_acc'] +val_accuracy

    # add 0th element, epochs should start from 1
    train_accuracy = ([logs['init_train_acc']] if 'init_train_acc' in logs else [0]) + train_accuracy
    val_accuracy = ([logs['init_val_acc']] if 'init_val_acc' in logs else [0]) + val_accuracy


    # summarize history for accuracy
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    if 'top_acc' in logs and 'top_val_acc' in logs and vertical_line:
        plt.axvline(vertical_line,color="red",linestyle='--')

    plt.grid()
    #plt.show()
    plt.savefig(file)
    plt.close()
    plt.clf()

def plot_loss_history(logs, file, vertical_line=None):
    plt.figure()

    train_loss = []
    val_loss = []

    if 'loss' in logs:
        train_loss = logs['loss']
    if 'val_loss' in logs:
        val_loss = logs['val_loss']

    if 'top_loss' in logs and 'top_val_loss' in logs and vertical_line:
        train_loss = logs['top_loss'] + train_loss
        val_loss = logs['top_val_loss'] + val_loss

    # add 0th element, epochs should start from 1
    train_loss = ([logs['init_train_loss']] if 'init_train_loss' in logs else [0]) + train_loss
    val_loss = ([logs['init_val_loss']] if 'init_val_loss' in logs else [0]) + val_loss

    # summarize history for lossuracy
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    if 'top_loss' in logs and 'top_val_loss' in logs and vertical_line:
        plt.axvline(vertical_line, color="red", linestyle='--')

    plt.grid()
    #plt.show()
    plt.savefig(file)
    plt.close()
    plt.clf()

def plot_class_acc_history(class_index, logs, file,vertical_line=None):
    plt.figure()

    train_accuracy = numpy.array(logs['train_per_class'])[:, class_index]
    val_accuracy = numpy.array(logs['val_per_class'])[:, class_index]

    train_accuracy = []
    val_accuracy = []

    if 'train_per_class' in logs:
        train_accuracy = logs['train_per_class']

    if 'val_per_class' in logs:
        val_accuracy = logs['val_per_class']

    if 'top_train_per_class' in logs and 'top_val_per_class' in logs and vertical_line:
        train_accuracy = numpy.concatenate((numpy.array(logs['train_per_class'])[:, class_index],numpy.array(logs['top_train_per_class'])[:, class_index]))
        val_accuracy = numpy.concatenate((numpy.array(logs['val_per_class'])[:, class_index],numpy.array(logs['top_val_per_class'])[:, class_index]))

    # summarize history for accuracy
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    if 'top_acc' in logs and 'top_val_acc' in logs and vertical_line:
        plt.axvline(vertical_line,color="red",linestyle='--')

    plt.grid()
    #plt.show()
    plt.savefig(file)
    plt.close()
    plt.clf()

def print_cmax(file, cmat, results):
    with open(file, "w") as f:
        for row in cmat:
            for col in row:
                f.write("{0:3d}\t".format(col))
                print("{0:3d}\t".format(col))
            f.write("\n")
            print("")
        f.write(str(results) + "\n")
        print(results)
        f.write(str(np.sum(cmat)))
        print(np.sum(cmat))


def main():
    data_dir = '/mnt/sda2/dev_root/dataset/FER13-Processed/small/'
    model_dir = '/mnt/sda2/dev_root/work/inception/inception_1/checkpoints/m.hdf5'
    weights_dir = '/mnt/sda2/dev_root/work/inception/inception_1/checkpoints/w.030-0.57-2.03.hdf5'
    model = load_model(model_dir)
    model.load_weights(weights_dir)
    dimension = (224, 224)
    batch_size = 1

    generate_cmax(data_dir, model, dimension, batch_size, os.path.dirname(weights_dir))

    return

def read_history_from_csv_file(file_path):
    with open(file_path, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    headers = data[0]
    data=numpy.array(data[1:])

    result={}

    for i,header in enumerate(headers):
        result[header]=data[:,i]

    return result

if __name__ == '__main__':
    main()

    # os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")
