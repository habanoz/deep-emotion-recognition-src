"""You can download a trained facial shape predictor from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)"""

import dlib
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os

predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor_path = "/home/habanoz/Downloads/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
faces_image_path = "iamages/00000005.png"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Faces-Structured/Val/Angry/001855040_I_1032.jpg"


def detect_landmarks(image):

    # obtain detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # convert image to numpy array
    img = np.asanyarray(image)
    img.flags.writeable =True

    #output list
    face_landmark_tuples=[]

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, rect in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
              .format(k, rect.left(), rect.top(), rect.right(), rect.bottom()))

        # Get the landmarks/parts for the face in box rect.
        shape = predictor(img, rect)
        face_landmark_tuples.append((k,rect,shape))

    return face_landmark_tuples

def draw_landmarks(image,shape):
    parts = []
    radius=1

    # copy original image, do not touch it
    out_image=image.copy()

    # extract parts to list
    for i in xrange(0, shape.num_parts, 1):
        print("Part {}".format(shape.part(i)))
        parts.append(shape.part(i))

    # for each part, draw a circle
    draw = ImageDraw.Draw(out_image)
    for part in parts:
        x = part.x
        y = part.y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0, 255))

    return out_image


def draw_glasses(image,shape):

    # obtain image dimensions and calculate offsets
    width,height=image.size
    offset_vertical=height/50
    offset_horizontal=width/50

    # add left and right glass points
    parts_left=obtain_left_glass(offset_horizontal, offset_vertical, shape)
    parts_right=obtain_right_glass(offset_horizontal, offset_vertical, shape)

    # copy original image, do not touch at
    out_image=image.copy()

    # draw lines
    draw = ImageDraw.Draw(out_image)
    draw.line(parts_left, fill=(0, 0, 0), width=10)
    draw.line(parts_right, fill=(0, 0, 0), width=10)

    return out_image


def obtain_right_glass(offset_horizontal, offset_vertical, shape):
    parts_right=[]

    # add tip of nose
    part = shape.part(28)
    _x = part.x + 2 * offset_horizontal
    _y = part.y
    parts_right.append((_x, _y))

    # add eyebrow
    for i in xrange(22, 27, 1):
        _x = shape.part(i).x
        _y = shape.part(i).y - offset_vertical
        parts_right.append((_x, _y))

    # right top of face
    part = shape.part(16)
    _x = part.x
    _y = part.y - offset_vertical
    parts_right.append((_x, _y))

    # right nostril
    part = shape.part(35)
    _x = part.x + 4 * offset_horizontal
    _y = part.y - 4 * offset_vertical
    parts_right.append((_x, _y))

    # add tip of nose again to close loop
    part = shape.part(28)
    _x = part.x + 2 * offset_horizontal
    _y = part.y
    parts_right.append((_x, _y))

    return parts_right


def obtain_left_glass(offset_horizontal, offset_vertical, shape):
    parts_left = []

    # left top of face
    part = shape.part(0)
    _x = part.x
    _y = part.y + offset_vertical
    parts_left.append((_x, _y))

    # add eyebrow
    for i in xrange(17, 22, 1):
        _x = shape.part(i).x
        _y = shape.part(i).y - offset_vertical
        parts_left.append((_x, _y))

    # add tip of nose
    part = shape.part(28)
    _x = part.x - 2 * offset_horizontal
    _y = part.y
    parts_left.append((_x, _y))

    # left nostril
    part = shape.part(31)
    _x = part.x - 4 * offset_horizontal
    _y = part.y - 4 * offset_vertical
    parts_left.append((_x, _y))

    # left top of face to close the loop
    part = shape.part(0)
    _x = part.x
    _y = part.y + offset_vertical
    parts_left.append((_x, _y))

    return parts_left


def main():
    # read image and convert to np array
    image=Image.open(faces_image_path)
    # detect faces and landmarks
    face_landmark_tuples = detect_landmarks(image=image)

    for k,rect,shape in face_landmark_tuples:
        # draw landmarks to image
        image=draw_landmarks(image,shape=shape)

        # determine file name and save new image
        path_name=os.path.dirname(faces_image_path)
        output_file=os.path.join(path_name,str(k)+'asd'+os.path.basename(faces_image_path))

    image.show()

if __name__ =='__main__':
    main()