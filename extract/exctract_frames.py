import os

from extract.DataProcessBase import DataProcessBase

# constants
FFMPEG_INPUT_SUFFIX = '.avi'
FFMPEG_OUTPUT_SUFFIX = '-%03d.jpg'
FIRST_SAMPLE_SUFFIX = '-001.jpg'


class ProcessVideos(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None, generate_data_file_only=False):
        super(ProcessVideos,self).__init__(source_dir,target_dir, data_file_index, dimension, limit_input_dirs, generate_data_file_only)
        self.process_description='Extracting Frames'

    def do_process(self, source_row_tuple):
        # un-box row to variables
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_row_tuple

        if not ProcessVideos.check_already_extracted(source_row_tuple, self.target_dir):
            # Now extract it.
            src = self.source_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + FFMPEG_INPUT_SUFFIX
            dest = self.target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + FFMPEG_OUTPUT_SUFFIX

            # create high quality frames
            os.system("ffmpeg -i " + src + " -qscale:v 2 " + dest + " >> " + self.target_dir + '/log.txt 2>&1')

        return

    @staticmethod
    def check_already_extracted(video_parts, target_dir):
        """Check to see if we created the -001 frame of this file."""
        input_dir, class_name, filename_no_ext, _ = video_parts

        return bool(os.path.exists(
            target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + FIRST_SAMPLE_SUFFIX))
