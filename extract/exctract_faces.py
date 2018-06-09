import glob
import os

import cv2
import mxnet

from extract.DataProcessBase import DataProcessBase
from extract.exctract_frames import FIRST_SAMPLE_SUFFIX
from thirdp.mxnet_mtcnn.mtcnn_detector import MtcnnDetector

# constants

detector = MtcnnDetector(model_folder='thirdp/mxnet_mtcnn/model', ctx=mxnet.cpu(0), num_worker=4, accurate_landmark=False)

class ExtractFaces(DataProcessBase):
    def __init__(self, source_dir, target_dir, data_file_index=0, dimension=224, limit_input_dirs=None, generate_data_file_only=False,align_faces=None,skip_existing=False):
        super(ExtractFaces, self).__init__(source_dir, target_dir, data_file_index, dimension, limit_input_dirs, generate_data_file_only)
        self.process_description='Extracting Faces'
        self.align_faces=align_faces
        self.skip_existing=skip_existing

    def do_process(self, source_row_tuple):
        # un-box row to variables
        input_dir, class_name, filename_no_ext, nb_sub_samples = source_row_tuple

        # list frame files
        frame_files = glob.glob( self.source_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + '*.*')

        # ensure files are sorted
        frame_files=sorted(frame_files)

        assert len(frame_files) != nb_sub_samples, "For sample {} sub-sample count does not match".format(filename_no_ext)

        # reset face images count
        nb_face_images = 0

        for frame_file in frame_files:

            frame_img = cv2.imread(frame_file)

            # save the image
            face_img_file = self.target_dir + '/' + input_dir + '/' + class_name + '/' + os.path.basename(frame_file)

            if not face_img_file.endswith('.jpg'):
                face_img_file = face_img_file.replace('.png', '.jpg')
                face_img_file = face_img_file.replace('.jpeg', '.jpg')
                face_img_file = face_img_file.replace('.gif', '.jpg')

            if self.skip_existing and os.path.exists(face_img_file):
                return

            # detect face
            results = detector.detect_face(frame_img)

            # if no faces detected continue with next frame file
            if results is None:
                continue

            total_boxes = results[0]
            points = results[1]


            if self.align_faces:
                # extract aligned face chips
                aligned_faces = detector.extract_image_chips(frame_img, points, self.dimension[0], 0.1)

                # use only largest face
                aligned_face = max(aligned_faces, key=lambda rect: rect.shape[0] * rect.shape[1])

                # increase face images count, in other words frames of the video containing at least a face
                nb_face_images = nb_face_images + 1

                face_img=aligned_face
            else:
                # use largest box  top lef (x,y) bottom right (t,u) => (t-x)*(u-y) is area
                # x=b[0], y=b[1], t=b[2], u=b[3]
                # b[4] gives score which is more reliable

                box = max(total_boxes, key=lambda b:b[4])

                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])


                x2 = min(max(0, x2), frame_img.shape[1])
                y2 = min(max(0, y2), frame_img.shape[0])
                x1 = min(max(0, x1), x2)
                y1 = min(max(0, y1), y2)

                face_img = frame_img[y1:y2,x1:x2]
                face_img = cv2.resize(face_img,self.dimension)

            cv2.imwrite(face_img_file, face_img)

        return

    @staticmethod
    def check_already_extracted(video_parts, target_dir):
        """Check to see if we created the -001 frame of this file."""
        input_dir, class_name, filename_no_ext, _ = video_parts

        return bool(os.path.exists(
            target_dir + '/' + input_dir + '/' + class_name + '/' + filename_no_ext + FIRST_SAMPLE_SUFFIX))
