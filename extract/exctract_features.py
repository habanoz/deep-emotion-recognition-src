import glob
import os

import numpy as np
from thirdp.harvitronix.extract.data import DataSet
from thirdp.harvitronix.extract.extractor import Extractor

from extract.DataProcessBase import DataProcessBase

# constants
FEATURE_FILE_SUFFIX = '-f.txt'


class ExtractFeatures(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None,
                 generate_data_file_only=False, seq_length=40, pretrained_model=None, layer_name=None):

        super(ExtractFeatures, self).__init__(source_dir, target_dir, data_file_index, dimension, limit_input_dirs,
                                              generate_data_file_only)

        self.process_description = 'Extracting Features'
        self.seq_length = seq_length

        # get the model.
        self.extractor = Extractor(pretrained_model, layer_name, (dimension, dimension))

        # Get the dataset.
        self.data = DataSet(source_dir + '/data.csv', target_dir, seq_length=seq_length, class_limit=None)

    def do_process(self, source_row_tuple):
        # un-box row to variables
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_row_tuple

        if int(nb_sub_samples) < self.seq_length:
            return

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
        sub_samples = self.data.rescale_list(sub_samples, self.seq_length)

        sequence = []
        for sub_sample in sub_samples:
            # extract features to build the sequence.
            features = self.extractor.extract(sub_sample)
            sequence.append(features)

        # Save the sequence.
        np.savetxt(path, np.array(sequence).reshape((self.seq_length, -1)))

        return

    def get_nb_sub_samples(self, sample_tuple):
        """
        Return generated number of sub samples for the sample.

        :param sample_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return: number of sub samples
        """

        # if file exists we know that there are exactly seq_length sub samples inside it.
        # if file does not exits we can say that there are no sub samples
        if ExtractFeatures.check_already_extracted(sample_tuple, self.target_dir):
            return self.seq_length

        return 0

    @staticmethod
    def check_already_extracted(sample_tuple, target_dir):
        """Check to see if we created the -001 frame of this file."""
        input_dir, class_name, filename_no_ext, _ = sample_tuple

        return bool(os.path.exists(
            target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + FEATURE_FILE_SUFFIX))
