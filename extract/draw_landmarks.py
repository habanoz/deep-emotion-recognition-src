import glob
import os

import dlib
import numpy as np
from PIL import Image
from PIL import Image
from PIL import ImageDraw
from thirdp.harvitronix.extract.data import DataSet

from extract.DataProcessBase import DataProcessBase

# constants
PREDICTOR_PATH = "face/dlib-models/shape_predictor_68_face_landmarks.dat"
LM_SUFFIX='-lm.jpg'

class DrawLandmarks(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None,
                 generate_data_file_only=False, seq_length=40, pretrained_model=None, layer_name=None):

        super(DrawLandmarks, self).__init__(source_dir, target_dir, data_file_index, dimension, limit_input_dirs,
                                               generate_data_file_only)

        self.process_description = 'Drawing Land Marks'
        self.seq_length = seq_length

        # obtain detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        # Get the dataset.
        self.data = DataSet(source_dir + '/data.csv', target_dir, seq_length=seq_length, class_limit=None)

    def do_process(self, source_row_tuple):
        # un-box row to variables
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_row_tuple

        if int(nb_sub_samples) < self.seq_length:
            return

        # Get the path to the sequence for this sub sample.
        target_class_path = self.target_dir + '/' + input_dir + '/' + class_name + '/'

        if not os.path.exists(target_class_path):
            os.makedirs(target_class_path)

        # list sub-samples
        sub_samples = glob.glob(self.source_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        # Now downs ample to just the ones we need.
        if self.seq_length:
            sub_samples = self.data.rescale_list(sub_samples, self.seq_length)

        for sub_sample in sub_samples:


            sub_sample_img = Image.open(sub_sample)
            # extract features to build the sequence.
            landmarks = self.__detect_landmarks(sub_sample_img)
            if landmarks:
                sub_sample_img_lm=self.draw_landmarks(sub_sample_img,landmarks)
                sub_sample_img_lm.save(target_class_path+os.path.splitext(os.path.basename(sub_sample))[0]+LM_SUFFIX)



        return

    def get_nb_sub_samples(self, sample_tuple):
        """
        Return generated number of sub samples for the sample.

        :param sample_tuple: has the structure input_dir, class_name, filename_no_ext, nb_sub_samples
        :return: number of sub samples
        """
        input_dir, class_name, filename_no_ext, nb_sub_samples = sample_tuple
        sub_samples = glob.glob(self.target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        return len(sub_samples)

    def __detect_landmarks(self, sub_sample_img):


        # convert image to numpy array
        img = np.asanyarray(sub_sample_img)
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

    def draw_landmarks(self, image, parts):
        radius = 1

        # copy original image, do not touch it
        out_image = image.copy()

        if out_image.mode != "RGB":
            out_image = out_image.convert("RGB")

        # for each part, draw a circle
        draw = ImageDraw.Draw(out_image)
        for part in parts[0]:
            x = part.x
            y = part.y
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(250,0,0))

        return out_image

