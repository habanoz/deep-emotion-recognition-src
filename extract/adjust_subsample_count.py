import glob
import os

import cv2
from PIL import Image

from thirdp.harvitronix.extract.data import DataSet

from extract.DataProcessBase import DataProcessBase

# constants
FEATURE_FILE_SUFFIX = '-f.txt'


class AdjustSubsampleCount(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None,
                 generate_data_file_only=False, seq_length=40, use_padding=False, nb_min_subsample=None):

        super(AdjustSubsampleCount, self).__init__(source_dir, target_dir, data_file_index, dimension, limit_input_dirs,
                                                   generate_data_file_only)

        self.process_description = 'Adjusting Subsample Count to '+str(seq_length)
        self.seq_length = seq_length


        # Get the dataset.
        self.data = DataSet(source_dir + '/data.csv', target_dir, seq_length=seq_length, class_limit=None)

        self.use_padding=use_padding
        self.nb_min_subsample = seq_length

        if use_padding:
            if nb_min_subsample:
                self.nb_min_subsample=nb_min_subsample
            else:
                self.nb_min_subsample=seq_length/2

    def do_process(self, source_row_tuple):
        # un-box row to variables
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_row_tuple

        if int(nb_sub_samples) < self.nb_min_subsample:
            return


        # if nb sub samples less than seq length, find required padding sub sample count
        # if padding not required, e.g. there are enough or more that enoguh samples, count is zero
        # if padding not enabled and padding is needed, previous statement does not allow reaching this statement
        nb_padding_needed=max(0,self.seq_length - int(nb_sub_samples))

        # Get the path to the sequence for this sub sample.
        path = self.target_dir + '/' + input_dir + '/'+ class_name + '/' + os.path.basename(filename_no_ext) + FEATURE_FILE_SUFFIX

        # Check if we already have it.
        if os.path.isfile(path):
            return

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # list sub-samples
        sub_samples = glob.glob(self.source_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        # Now downs ample to just the ones we need.
        if nb_padding_needed==0:
            sub_samples = self.data.rescale_list(sub_samples, self.seq_length)

        for i in range(nb_padding_needed):
            empty_image = Image.new('RGB', self.dimension)
            target_file = self.target_dir + '/' + input_dir + '/' + class_name + '/' + os.path.basename(filename_no_ext)+'_'+'{:08d}'.format(0)+'_'+'{:02d}'.format(i)+'.jpg'
            empty_image.save(target_file)

        for sub_sample in sub_samples:
            # extract features to build the sequence.
            img=cv2.imread(sub_sample)
            height, width = img.shape[:2]
            if self.dimension[0]!=height or self.dimension[1]!=width:
                img = cv2.resize(img,self.dimension)

            target_file=self.target_dir + '/' + input_dir + '/' + class_name + '/' + os.path.basename(sub_sample)
            cv2.imwrite(target_file,img)

        return

    def get_nb_sub_samples(self, sample_tuple):
        """
        Return generated number of sub samples for the sample.

        :param sample_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return: number of sub samples
        """
        input_dir, class_name, filename_no_ext, _ = sample_tuple
        sub_samples = glob.glob(self.target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        return len(sub_samples)
