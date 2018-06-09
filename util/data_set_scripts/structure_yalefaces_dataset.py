import os
import csv
import glob
import shutil
from PIL import Image

from util.directory_utils import get_path_parts




def structure(source_root, target_root):
    """
    Yale face data set is inform of 
    
    subject01.gif
    subject01.glasses
    subject01.glasses.gif
    subject01.happy
    subject01.leftlight
    subject01.noglasses
    subject01.normal
    subject01.rightlight
    subject01.sad
    subject01.sleepy
    subject01.surprised
    subject01.wink

    Use only records ending with .happy, .sad .normal and .surprised Train/Val split,  move them to corresponding Happy,
    Sad, Neutral and Surprise folders, and create data file.

    :return:
    """

    target_classes=['Happy','Sad','Surprise','Neutral']
    source_extensions=['.happy','.sad','.surprised','.normal']

    if not os.path.exists(target_root):
        os.makedirs(target_root)


    data_file = []

    train = []
    val = []
    train_val_ratio = 0.6

    for extension in source_extensions:
        samples = glob.glob(source_root + '/*' + extension)
        nb_samples = len(samples)
        last_train_index = int(nb_samples * train_val_ratio)
        train=train+samples[0:last_train_index]
        val=val+samples[last_train_index:]

    print("Train samples {} test samples {}".format(len(train),len(val)))

    input_folder='Train'
    process_samples(data_file, input_folder, target_root, train)

    input_folder = 'Val'
    process_samples(data_file, input_folder, target_root, val)

    with open(target_root+'/'+'data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def process_samples(data_file, input_folder, target_root, train):
    for sample in train:
        _class = None
        if sample.endswith('.happy'):
            _class = 'Happy'
        if sample.endswith('.sad'):
            _class = 'Sad'
        if sample.endswith('.surprised'):
            _class = 'Surprise'
        if sample.endswith('.normal'):
            _class = 'Neutral'

        filename_no_ext = os.path.splitext(os.path.basename(sample))[0]
        target_class_dir=target_root + '/' + input_folder + '/' + _class
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        target_sample = target_class_dir + '/' + filename_no_ext + '.jpg'

        img = Image.open(sample)
        img.save(target_sample)

        print("Source File {} processed".format(sample))

        data_file.append([input_folder, _class, filename_no_ext, 1])


def main():
    data_src_root = '/mnt/sda2/dev_root/dataset/original/yalefaces/yalefaces'
    data_dest_root = '/mnt/sda2/dev_root/dataset/original/yalefaces-structured'

    structure(data_src_root,data_dest_root)

if __name__ == '__main__':
    main()
