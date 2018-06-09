import glob
import os

import cv2

from extract.DataProcessBase import DataProcessBase


# constants

class PreprocessFaces(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None,
                 generate_data_file_only=False):
        super(PreprocessFaces, self).__init__(source_dir, target_dir, data_file_index, dimension, limit_input_dirs,
                                              generate_data_file_only)
        self.process_description = 'Preprocessing Images'

    def do_process(self, source_row_tuple):
        # un-box row to variables
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_row_tuple

        # list frame files
        subsamples = glob.glob(self.source_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        # ensure files are sorted
        subsamples = sorted(subsamples)

        assert len(subsamples) != nb_sub_samples, "For sample {} sub-sample count does not match".format(filename_no_ext)

        # reset face images count
        nb_face_images = 0

        for subsample in subsamples:

            img = cv2.imread(subsample, cv2.IMREAD_GRAYSCALE)

            height, width = img.shape[:2]

            # if dimensions are not as desired, resize the image
            if self.dimension != height or self.dimension != width:
                img = cv2.resize(img, self.dimension, interpolation=cv2.INTER_CUBIC)

            # Histogram Equalization
            img = cv2.equalizeHist(img)

            # save the image
            processed_subsample = self.target_dir + '/' + input_dir + '/' + class_name + '/' + os.path.basename(
                subsample)

            if not processed_subsample.endswith('.jpg'):
                processed_subsample = processed_subsample.replace('.png', '.jpg')
                processed_subsample = processed_subsample.replace('.jpeg', '.jpg')
                processed_subsample = processed_subsample.replace('.gif', '.jpg')

            cv2.imwrite(processed_subsample, img)

        return
