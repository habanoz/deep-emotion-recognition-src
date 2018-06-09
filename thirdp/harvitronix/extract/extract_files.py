"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

from util.directory_utils import get_path_parts


def extract_files(input_folders = ['./train/', './test/'],output_root='out/', data_file_path='data_file.csv'):
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """

    # create output_root dir if necessary
    if not os.path.exists(output_root):
        os.makedirs(output_root)


    data_file = []

    for folder in input_folders:
        class_folders = glob.glob(folder + '*')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.avi')

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_path_parts(video_path)

                root, train_or_test, classname, filename_no_ext, filename = video_parts
                extracted_frames_parts = output_root, train_or_test, classname, filename_no_ext, filename

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not __check_already_extracted(extracted_frames_parts):
                    # Now extract it.
                    src = root + '/' +train_or_test + '/' + classname + '/' + filename
                    dest = output_root + '/' +train_or_test + '/' + classname + '/' + filename_no_ext + '-%03d.jpg'

                    # create output_root dir if necessary
                    if not os.path.exists(os.path.dirname(dest)):
                        os.makedirs(os.path.dirname(dest))

                    call(["ffmpeg", "-i", src,"-qscale:v 2", dest])

                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(extracted_frames_parts)

                data_file.append([output_root, train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open(data_file_path, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    root, train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(root + '/' + train_or_test + '/' + classname + '/' +
                                filename_no_ext + '*.jpg')
    return len(generated_files)


def __check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    root, train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(root + '/' + train_or_test + '/' + classname +
                               '/' + filename_no_ext + '-0001.jpg'))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    input_folders = ['./train/', './test/']
    extract_files(input_folders)

if __name__ == '__main__':
    main()
