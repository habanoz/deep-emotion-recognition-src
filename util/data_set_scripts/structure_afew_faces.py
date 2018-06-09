import os
import csv
import glob
import shutil
from PIL import Image

from util.directory_utils import get_path_parts




def structure_faces(afew_samples_root, afew_samples_folders, afew_faces_root, target_afew_faces_root,convert2RGB=True,resize=True,size=(150,150)):
    """
    Faces data coming from AFEW bundle is inform of video_name/type/frame_number.jpg strcuture.
    Convert it into type/class/video_name_frame_number.jpg structure and create a data csv file.

    :return:
    """

    if not os.path.exists(target_afew_faces_root):
        os.makedirs(target_afew_faces_root)

    data_file = []

    for folder in afew_samples_folders:
        class_folders = [x  for x  in glob.glob(afew_samples_root + '/' + folder+'/*') if os.path.isdir(x)]

        for class_folder in class_folders:
            """
            We will learn class of face images by looking at class folder of video data
            """
            class_samples = glob.glob(class_folder + '/*.avi')

            for sample in class_samples:
                root, train_or_test, classname, filename_no_ext, filename = get_path_parts(sample)

                sample_face_image_folder = afew_faces_root + '/'+ train_or_test + '/' + filename_no_ext
                sample_face_images = glob.glob(sample_face_image_folder +'/'+ '*.jpg')

                target_face_folder=target_afew_faces_root + '/' + train_or_test + '/' + classname

                if not os.path.exists(target_face_folder):
                    os.makedirs(target_face_folder)

                # copy sample faces to target folder
                for sample_face_image in sample_face_images:
                    _, _, _, face_filename_no_ext, _ = get_path_parts(sample_face_image)


                    target_face_image=target_face_folder + '/' + filename_no_ext + '_' + face_filename_no_ext + '.jpg'

                    img=Image.open(sample_face_image)
                    if resize:
                        img = img.resize(size, Image.ANTIALIAS)

                    if convert2RGB:
                        img = img.convert('RGB')

                    img.save(target_face_image)


                print("copied faces for sample %s" % (sample_face_image_folder))

                data_file.append(
                    [target_afew_faces_root, train_or_test, classname, filename_no_ext, len(sample_face_images)]
                )

    with open(target_afew_faces_root+'/'+'data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def main():
    afew_samples_root = '/mnt/sda2/dev_root/dataset/AFEW_6_2016'
    afew_faces_root = '/mnt/sda2/dev_root/dataset/AFEW-Faces'
    target_afew_faces_root = '/mnt/sda2/dev_root/dataset/AFEW-Faces-Structured'
    afew_samples_folders = ['Train', 'Val']

    structure_faces(afew_samples_root,afew_samples_folders,afew_faces_root,target_afew_faces_root)

if __name__ == '__main__':
    main()
