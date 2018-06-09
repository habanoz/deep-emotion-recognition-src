from shutil import copyfile

"""
Generate data file for AFEW data set, so that we can work on the data set.
"""
import csv
import glob
import os
import os.path

from util.directory_utils import get_path_parts


def generate_data_file(data_set_root_directory, input_folders = ['Train', 'Val']):
    """
    [train|test], class, filename, nb frames
    """

    data_file = []

    for folder in input_folders:

        class_folders = glob.glob(data_set_root_directory+'/'+folder+'/' + '*')

        for _class in class_folders:
            class_files = glob.glob(_class + '/*.avi')

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_path_parts(video_path)

                root, train_or_test, classname, filename_no_ext, filename = video_parts

                # since we are working on videos, number of sub samples = 1
                nb_sub_samples = 1

                data_file.append([train_or_test, classname, filename_no_ext, nb_sub_samples])

    with open(data_set_root_directory+'/'+'data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def main():
    source_root='/mnt/sda2/dev_root/dataset/AFEW_6_2016'

    generate_data_file(source_root)

if __name__ == '__main__':
    main()
