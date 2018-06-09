import glob
import os
import abc
import csv
from time import sleep

from tqdm import tqdm

# constants
DATA_FILE_NAME = 'data.csv'

"""
This is a base class for all processor classes. This will include common code.
"""


class DataProcessBase(object):
    __metaclass__ = abc.ABCMeta

    process_description='Processing'

    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None,
                 generate_data_file_only=False):

        """
        Create a processor instance.

        :param source_dir: Directory where raw data files are present. This directory mush have structure
            [input_dir(Train,Val etc...)]/class_dir/samples. Some implementations may require presence of a source data
            file (data.csv) at this location.
        :param target_dir: Directory where processed data files will be put. A data.csv file will be created at this
            location.
        :param data_file_index: Index into source data file. By using this index, a previous un-finished processing can
            be continued. By default this is 0, which means start file from the beginning.
        :param dimension: Desired dimensions of the processed sample files. Not all processing implementations need to
            honor this. If single dimension is given converted to tuple. If image is not square specify with and height as tuple.
        :param limit_input_dirs: Limit processing to only a subset of input directories: Train, Test, Val, etc.
        :param generate_data_file_only: If set to True, skips processing and only re-creates the target data file.
            Default value is False. Set to True if only processing is already done once.
        """

        if not source_dir:
            raise Exception("Source data directory cannot be empty")

        if not target_dir:
            raise Exception("Target data directory cannot be empty")

        if isinstance(dimension, tuple):
            self.dimension = dimension
        else:
            self.dimension = (dimension,dimension)

        self.data_file_index = data_file_index
        self.limit_input_dirs = limit_input_dirs
        self.generate_data_file_only = generate_data_file_only
        self.source_dir = source_dir
        self.target_dir = target_dir

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

        # source data file path
        source_data_file_path = self.source_dir + '/'+ DATA_FILE_NAME

        # Get the data set.
        src_data_file_rows = self.__get_data(source_data_file_path, self.data_file_index, self.limit_input_dirs)

        # run processing inside progress bar block
        with tqdm(desc=self.process_description, total=len(src_data_file_rows), unit='it', unit_scale=False) as pbar:
            for src_row in src_data_file_rows:
                # un-box row to variables
                input_dir, class_name, filename_no_ext, nb_sub_samples = src_row

                class_directory = self.target_dir + '/' + input_dir + '/' + class_name

                # make sure the class folder is present
                if not os.path.exists(class_directory):
                    os.makedirs(class_directory)

                # run actual processing code, which is implementation specific
                self.do_process(src_row)

                pbar.update(1)

        sleep(0.1)
        print("Completed processing")

        # generate target data file
        self.__generate_target_data_file(self.source_dir, self.target_dir)

        return

    @abc.abstractmethod
    def do_process(self, row_tuple):
        """
        Implementation specific processing code

        :param row_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return:
        """
        return

    def get_nb_sub_samples(self, sample_tuple):
        """
        Return generated number of sub samples for the sample.

        :param sample_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return: number of sub samples
        """
        input_dir, class_name, filename_no_ext, _ = sample_tuple
        generated_files = glob.glob(self.target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.jpg')

        return len(generated_files)

    def __generate_target_data_file(self, source_dir, target_dir):

        # source data file path
        source_data_file_path = source_dir +'/'+ DATA_FILE_NAME

        # target data file path
        target_data_file_path = target_dir +'/'+ DATA_FILE_NAME

        print("Generating data file")
        print("Source data file is {}".format(source_data_file_path))
        print("Target data file is {}".format(target_data_file_path))
        sleep(0.1)

        # Get the data set.
        src_data_file_rows = self.__get_data(source_data_file_path, 0, None)

        # list for storing
        dst_data_file_rows = []

        # run file generation inside progress bar block
        with tqdm(desc='Generating Data File', total=len(src_data_file_rows), unit='it', unit_scale=False) as pbar:
            for src_row in src_data_file_rows:
                # un-box row to variables
                input_dir, class_name, filename_no_ext, nb_sub_samples = src_row

                # count processed sub samples for current sample, e.g. a video sample can be processed and frames are
                # created as sub-samples
                nb_processed_sub_samples = self.get_nb_sub_samples((input_dir, class_name, filename_no_ext, nb_sub_samples))

                dst_data_file_rows.append([input_dir, class_name, filename_no_ext, nb_processed_sub_samples])

                pbar.update(1)

        sleep(0.1)

        # write faces data to faces file
        with open(target_data_file_path, 'w') as fout:
            writer = csv.writer(fout)
            writer.writerows(dst_data_file_rows)

        print("Data file generated")

    def __get_data(self, data_file, start_idx, limit_input_dirs):
        """Load our data from csv file."""
        with open(data_file, 'r') as fin:
            reader = csv.reader(fin)
            samples = list(reader)

        # adjust to start index, typically from beginning
        samples = samples[start_idx:]

        if limit_input_dirs:
            out_samples = []
            for sample in samples:
                input_dir, _, _, _ = sample
                if input_dir not in limit_input_dirs:
                    continue

                out_samples.append(sample)

            samples = out_samples

        return samples
