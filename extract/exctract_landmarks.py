import glob
import os

import dlib
import numpy as np
from PIL import Image

from thirdp.harvitronix.extract.data import DataSet

from extract.DataProcessBase import DataProcessBase

# tracking only moving points should be enough
# face is symmetric, one half is enough
LEFT_EYEBROW_INDICES=list(range(17,22))
LEFT_EYE_INDICES=list(range(36,42))
LEFT_LIPS_INDICES=[48,49,50,51,57,58,59]
LEFT_MOUTH_INDICES=[60,61,62,66,67]

CENTER_OF_FACE_INDEX=33 # all points will be positioned relative to this point
LEFT_OF_FACE_INDEX=0    # face width left point
RIGHT_OF_FACE_INDEX=16  # face width right point
TOP_OF_FACE_INDEX=27    # face height top point, 27 is top of noise. We use it because only fixed point inside upper face
BOTTOM_OF_FACE_INDEX=33 # face height bottom point, 33 is bottom of noise also center of face. We use it because only fixed point inside lower face
FACE_HEIGHT_MULTIPLIER=2 # nose length is not enough as face length. Multiply nose length for a more reliable face length.

# constants
LM_FILE_SUFFIX = '-lm.txt.npy'
PREDICTOR_PATH = "face/dlib-models/shape_predictor_68_face_landmarks.dat"


class ExtractLandmarks(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None,
                 generate_data_file_only=False, seq_length=40, use_padding=False, nb_min_subsample=None):

        super(ExtractLandmarks, self).__init__(source_dir, target_dir, data_file_index, dimension, limit_input_dirs,
                                               generate_data_file_only)

        self.process_description = 'Extracting Land Marks'
        self.seq_length = seq_length

        # obtain detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        # Get the dataset.
        self.data = DataSet(source_dir + '/data.csv', target_dir, seq_length=seq_length, class_limit=None)

        self.use_padding = use_padding
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

        # Get the path to the sequence for this sub sample.
        path = self.target_dir + '/' + input_dir + '/' + class_name + '/' + os.path.basename(
            filename_no_ext) + LM_FILE_SUFFIX

        # Check if we already have it.
        if os.path.exists(path):
            return

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # list sub-samples
        sub_samples = glob.glob(self.source_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        # Now downs ample to just the ones we need.
        if self.seq_length:
            sub_samples = self.data.rescale_list(sub_samples, self.seq_length)

        sequence = []
        for sub_sample in sub_samples:
            # extract features to build the sequence.
            landmarks = self.__detect_landmarks(sub_sample)
            if landmarks:
                landmarks = self.__normalize_landmarks(landmarks)
                sequence.append(landmarks)

        if len(sequence) < self.seq_length and len(sequence) > self.nb_min_subsample and self.use_padding:
            nb_padding_needed = max(0, self.seq_length - len(sequence))
            for i in range(nb_padding_needed):
                sequence.append([0 for x in range(0, 23)]) # pad with zeros

        if len(sequence)==self.seq_length:
            # Save the sequence.
            np.save(path, np.array(sequence))

        return

    def get_nb_sub_samples(self, sample_tuple):
        """
        Return generated number of sub samples for the sample.

        :param sample_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return: number of sub samples
        """

        # if file exists we know that there are exactly seq_length sub samples inside it.
        # if file does not exits we can say that there are no sub samples
        if ExtractLandmarks.check_already_extracted(sample_tuple, self.target_dir):
            input_dir, class_name, filename_no_ext, _ = sample_tuple
            lm_arr = np.load(self.target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + LM_FILE_SUFFIX)
            return len(lm_arr)
        return 0

    def __detect_landmarks(self, image_path):

        image = Image.open(image_path)
        # convert image to numpy array
        img = np.asanyarray(image)
        img.flags.writeable = True

        # output list
        face_landmark_tuples = []

        # Obtain landmarks
        dets = self.detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for k, rect in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
                  .format(k, rect.left(), rect.top(), rect.right(), rect.bottom()))

            # Get the landmarks/parts for the face in box rect.
            shape = self.predictor(img, rect)
            face_landmark_tuples.append([shape.part(x) for x in range(68)])

        return face_landmark_tuples

    @staticmethod
    def check_already_extracted(sample_tuple, target_dir):
        """Check to see if we created the -001 frame of this file."""
        input_dir, class_name, filename_no_ext, _ = sample_tuple

        return bool(os.path.exists(target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + LM_FILE_SUFFIX))

    def __normalize_landmarks(self, landmarks):

        landmarks=landmarks[0]

        top_point = landmarks[TOP_OF_FACE_INDEX]
        bottom_point = landmarks[BOTTOM_OF_FACE_INDEX]
        heigth = FACE_HEIGHT_MULTIPLIER * (bottom_point.y - top_point.y)

        center_y=landmarks[CENTER_OF_FACE_INDEX].y

        lm_indices = []
        lm_indices.extend(LEFT_EYEBROW_INDICES)
        lm_indices.extend(LEFT_EYE_INDICES)
        lm_indices.extend(LEFT_LIPS_INDICES)
        lm_indices.extend(LEFT_MOUTH_INDICES)

        normalized_landmarks=[]
        for idx in lm_indices:
            lm_point_y=landmarks[idx].y
            normalized_landmarks.append( abs(lm_point_y-center_y)*1.0/heigth )


        return normalized_landmarks
