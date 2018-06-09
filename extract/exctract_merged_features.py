import csv
import glob
import os
from time import sleep

import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from extract.preprocess.merge_extractor import MergerExtractor
from extract.DataProcessBase import DataProcessBase, DATA_FILE_NAME

# constants
from util.generator.MergeCsvGenerator import MergeGenerator
from util.generator.SequenceFeatureCsvGenerator import SequenceFeatureIGenerator
from util.generator.SequenceImageCvsGenerator import SequenceImageIGenerator

FEATURE_FILE_SUFFIX = '-f.txt'


class ExtractMergedFeatures(DataProcessBase):
    """
    This specific implementation uses c3d and lstm models. Model files and data paths are not parametric to have a simple interface.


    """
    def __init__(self, source_dir, target_dir, data_file_index=0):

        super(ExtractMergedFeatures, self).__init__(source_dir, target_dir, data_file_index, None, None,False)

        self.process_description = 'Extracting Merged Features'

        model_files = ['/mnt/sda2/dev_root/work2/c3d/7t-c3d__16_112_adam_b8_1lr1e6/checkpoints/w.014-0.3507-2.81.hdf5',
                       '/mnt/sda2/dev_root/work2/lstm/2_lstm_40_224/checkpoints/w.029-0.4563-1.58.hdf5']

        csv_file_path_c3d = '/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_PFaces_16_112_small/data.csv'
        csv_file_path_lstm = '/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Features_vggface_fc7_224/data.csv'

        seq_length_lstm = 40
        seq_length_c3d = 16
        image_dim = 112

        # get the model.
        self.extractor = MergerExtractor(model_files)

        image_data_generator = ImageDataGenerator(rescale=1.0 / 255)
        image_generator = SequenceImageIGenerator(seq_length_c3d, image_data_generator, (image_dim, image_dim))

        feature__generator = SequenceFeatureIGenerator(seq_length_lstm)

        self.merged_generator_train = MergeGenerator([image_generator, feature__generator],[csv_file_path_c3d, csv_file_path_lstm], True, 1)
        self.merged_generator_valid = MergeGenerator([image_generator, feature__generator],[csv_file_path_c3d, csv_file_path_lstm], False, 1)

    def process(self):
        """Do common processing and delegate specific processing to do_process method"""

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        if self.generate_data_file_only:
            self.__generate_target_data_file(self.source_dir, self.target_dir)
            return

        print("Starting processing")
        print("Source dir is {}".format(self.source_dir))
        print("Target dir is {}".format(self.target_dir))
        sleep(0.1)

        # Get the data set.

        source_data=[]

        self.process_for_generator(source_data,self.merged_generator_train)
        self.process_for_generator(source_data,self.merged_generator_valid)

        print("Completed processing")

        # generate target data file
        self.__generate_target_data_file(source_data, self.target_dir)

        return

    def process_for_generator(self, source_data,generator):
        igenerator = generator.flow()
        # run processing inside progress bar block
        with tqdm(desc=self.process_description, total=len(generator.data), unit='it',
                  unit_scale=False) as pbar:
            for i in range(len(generator.data)):

                input_data = igenerator.next()[0] # only input needed,label not needed
                sample_row = generator.data[i]

                # un-box row to variables
                input_dir, class_name, filename_no_ext, nb_sub_samples = sample_row

                class_directory = self.target_dir + '/' + input_dir + '/' + class_name

                # make sure the class folder is present
                if not os.path.exists(class_directory):
                    os.makedirs(class_directory)

                # run actual processing code, which is implementation specific
                self.do_process([input_data, sample_row])

                source_data.append(sample_row)

                pbar.update(1)
        sleep(0.1)

    def do_process(self, source_data_with_sample_info):

        # un-box row to variables
        source_data=source_data_with_sample_info[0]
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_data_with_sample_info[1]

        # Get the path to the sequence for this sub sample.
        path = self.target_dir + '/' + input_dir + '/'+ class_name + '/' + os.path.basename(filename_no_ext) + FEATURE_FILE_SUFFIX

        # Check if we already have it.
        if os.path.isfile(path):
            return

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        sequence = self.extractor.extract(source_data)

        # Save the sequence.
        np.savetxt(path, np.array(sequence))

        return

    def get_nb_sub_samples(self, sample_tuple):
        """
        Return generated number of sub samples for the sample.

        :param sample_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return: number of sub samples
        """

        # if file exists we know that there are exactly seq_length sub samples inside it.
        # if file does not exits we can say that there are no sub samples
        if ExtractMergedFeatures.check_already_extracted(sample_tuple, self.target_dir):
            return 1

        return 0

    def __generate_target_data_file(self, source_data, target_dir):

        # target data file path
        target_data_file_path = target_dir + '/' + DATA_FILE_NAME

        print("Generating data file")
        print("Target data file is {}".format(target_data_file_path))
        sleep(0.1)


        # list for storing
        dst_data_file_rows = []

        # run file generation inside progress bar block
        with tqdm(desc='Generating Data File', total=len(source_data), unit='it', unit_scale=False) as pbar:
            for src_row in source_data:
                # un-box row to variables
                input_dir, class_name, filename_no_ext, nb_sub_samples = src_row

                # count processed sub samples for current sample, e.g. a video sample can be processed and frames are
                # created as sub-samples
                nb_processed_sub_samples = self.get_nb_sub_samples(
                    (input_dir, class_name, filename_no_ext, nb_sub_samples))

                dst_data_file_rows.append([input_dir, class_name, filename_no_ext, nb_processed_sub_samples])

                pbar.update(1)

        sleep(0.1)

        # write faces data to faces file
        with open(target_data_file_path, 'w') as fout:
            writer = csv.writer(fout)
            writer.writerows(dst_data_file_rows)

        print("Data file generated")

    @staticmethod
    def check_already_extracted(sample_tuple, target_dir):
        """Check to see if we created the -001 frame of this file."""
        input_dir, class_name, filename_no_ext, _ = sample_tuple

        return bool(os.path.exists(
            target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + FEATURE_FILE_SUFFIX))


def main():
    target_dir='/mnt/sda2/dev_root/dataset/combined/merged/lstmc3d/'

    extract_merged_features=ExtractMergedFeatures('Dummy', target_dir)
    extract_merged_features.process()

    return

if __name__ == '__main__':
    main()