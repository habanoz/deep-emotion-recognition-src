from shutil import copyfile
from tqdm import tqdm
import cv2

"""
Generate data file for fer2013 data set, so that we can work on the data set.
Files are copied to new target directory with a suffix.
"""
import csv
import glob
import os
import os.path

def generate_data_file(data_dirs, target_dir, new_dimension=None):
    """
    [train|test], class, filename, nb frames
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    data_file = []
    for data_dir in data_dirs:
        samples = get_data(data_dir + '/data.csv')
        with tqdm(desc="Merging data set {}".format(data_dir), total=len(samples), unit='it', unit_scale=False) as pbar:
            for sample in samples:
                copy(data_dir, data_file, new_dimension, sample, target_dir)

                pbar.update(1)

    with open(target_dir+ '/'+ 'data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Processed and wrote %d image files." % (len(data_file)))


def copy(data_dir_1, data_file, new_dimension, sample, target_dir):
    # Get the parts of the file.
    input_dir, class_name, filename_no_ext, nb_subsamp = sample
    sample_dir = data_dir_1 + '/' + input_dir + '/' + class_name + '/' + filename_no_ext

    nb_subsamp=int(nb_subsamp)
    if nb_subsamp == 0:
        return
    elif nb_subsamp==1:
        sub_samples = [sample_dir+".jpg"]
    else:
        sub_samples = glob.glob(sample_dir + '*')
        assert len(sub_samples)==nb_subsamp

    target_class_directory = target_dir + '/' + input_dir + '/' + class_name

    if not os.path.exists(target_class_directory):
        os.makedirs(target_class_directory)
    for sub_sample in sub_samples:
        if new_dimension:
            im = cv2.imread(sub_sample)
            try:
                im = cv2.resize(im, (new_dimension, new_dimension), interpolation=cv2.INTER_CUBIC)
            except cv2.error as e:
                print("Errror at image {}".format(sub_sample))
                raise e

            cv2.imwrite(target_class_directory + '/' + os.path.basename(sub_sample), im)
        else:
            copyfile(sub_sample, target_class_directory + '/' + os.path.basename(sub_sample))

    # since all fo them single images, number of frames = 1
    nb_frames = len(sub_samples)
    data_file.append([input_dir, class_name, filename_no_ext, nb_frames])


def get_data(data_file):
    """Load our data from file."""
    with open(data_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    source_root_1='/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'
    source_root_2='/mnt/sda2/dev_root/dataset/sfew/Aligned_PFaces_224_p01/'
    source_root_2='/mnt/sda2/dev_root/dataset/SFEW-Processed/SFEW_PFaces_224/'
    source_root_2='/mnt/sda2/dev_root/dataset/sfew/pfaces-not-aligned-224/'
    source_root_3='/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224_p01/'
    source_root_3='/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224/'
    source_root_3='/mnt/sda2/dev_root/dataset/CK+/pfaces-not-aligned-224/'
    source_root_4='/mnt/sda2/dev_root/dataset/yale-Processed/yale_PFaces_224'
    source_root_4='/mnt/sda2/dev_root/dataset/yale-Processed/not_aligned_yale_PFaces_224/'
    target_root='/mnt/sda2/dev_root/dataset/combined/FER-NOT-ALIGNED-SFEW-CKP-YALE-224'

    generate_data_file([source_root_1,source_root_2,source_root_3,source_root_4], target_root, new_dimension=224)

if __name__ == '__main__':
    main()