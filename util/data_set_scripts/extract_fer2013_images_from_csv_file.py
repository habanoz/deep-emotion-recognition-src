
import cv2

"""
prepare data set by extracting images from original csv file.
Test samples are not added to generated data.csv file.
"""
import csv
import os
import os.path
import pandas as pd
import numpy as np

EMOTION_IDX=0
PIXELS_IDX=1
USAGE_IDX=2

ORIGINAL_DIM=48

def generate_data_file(csv_file,data_set_target_directory, input_folders = ['Train', 'Val']):

    data = pd.read_csv(csv_file)
    class_folders=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    usage_input_folder_mapping={'Training':'Train','PublicTest':'Val','PrivateTest':'Test'}

    if not os.path.exists(data_set_target_directory):
        os.makedirs(data_set_target_directory)

    data_file = []

    for i in range(len(data)):
        sample = data.values[i]
        class_name = class_folders[sample[EMOTION_IDX]]
        input_dir=usage_input_folder_mapping[sample[USAGE_IDX]]
        filename_no_ext='{}_{:04d}'.format(class_name.upper(),i)

        target_class_directory=data_set_target_directory+ '/'+ input_dir+ '/'+ class_name
        if not os.path.exists(target_class_directory):
            os.makedirs(target_class_directory)

        #im = cv2.imread(sample)
        im = np.fromstring(sample[PIXELS_IDX], dtype=int, sep=" ").reshape((ORIGINAL_DIM, ORIGINAL_DIM))
        cv2.imwrite(target_class_directory+'/'+filename_no_ext+'.jpg',im)


        # since all fo them single images, number of frames = 1
        nb_frames = 1

        if input_dir in input_folders:
            data_file.append([input_dir, class_name, filename_no_ext, nb_frames])

    with open(data_set_target_directory+'/'+'data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Processed and wrote %d image files." % (len(data_file)))


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    csv_file='/mnt/sda2/dev_root/dataset/original/fer2013.csv'
    target_root='/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW'

    generate_data_file(csv_file, target_root)

if __name__ == '__main__':
    main()
