from shutil import copyfile

import cv2

"""
Generate data file for fer2013 data set, so that we can work on the data set.
Files are copied to new target directory with a suffix.
"""
import csv
import glob
import os
import os.path

from util.directory_utils import get_path_parts


def generate_data_file(data_set_root_directory,data_set_target_directory, input_folders = ['Train', 'Val'], new_dimension=None):
    """
    [train|test], class, filename, nb frames
    """

    if not os.path.exists(data_set_target_directory):
        os.makedirs(data_set_target_directory)

    data_file = []

    for folder in input_folders:

        class_folders = glob.glob(data_set_root_directory+'/'+folder+'/' + '*')

        for _class in class_folders:
            samples = glob.glob(_class + '/*.jpg')

            for sample in samples:
                # Get the parts of the file.
                _, input_dir, class_name, filename_no_ext, _ = get_path_parts(sample)

                target_class_directory=data_set_target_directory+ '/'+ input_dir+ '/'+ class_name
                if not os.path.exists(target_class_directory):
                    os.makedirs(target_class_directory)

                if new_dimension:
                    im = cv2.imread(sample)
                    im = cv2.resize(im,(new_dimension,new_dimension),interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(target_class_directory+'/'+os.path.basename(sample),im)
                else:
                    copyfile(sample, target_class_directory+ '/'+os.path.basename(sample))

                # since all fo them single images, number of frames = 1
                nb_frames = 1

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
    source_root='/mnt/sda2/dev_root/dataset/original/fer2013'
    target_root='/mnt/sda2/dev_root/dataset/FER13-Processed/FER13-224'

    generate_data_file(source_root, target_root, new_dimension=224)

if __name__ == '__main__':
    main()
