import os

from shutil import copy

import cv2
from tqdm import tqdm

from util.csv_util import get_data,write_data


def generate(data_src_root,data_dest_root,dimension,trans_format=False,trans_name=False):
    data_file=data_src_root+'/data.csv'

    size=(dimension,dimension)

    source_data=get_data(data_file)

    target_data_list=[]

    with tqdm(desc='Copy dataset', total=len(source_data), unit='it', unit_scale=False) as pbar:
        for sample in source_data:
            input_type, _class, file_name_no_ext, nb_subsample = sample

            nb_subsample = int(nb_subsample)

            if nb_subsample > 1:
                raise Exception('Video data not supported!')

            if nb_subsample == 0:
                raise Exception('No sample!')


            if not face_img_file.endswith('.jpg'):
                face_img_file = face_img_file.replace('.png', '.jpg')
                face_img_file = face_img_file.replace('.jpeg', '.jpg')
                face_img_file = face_img_file.replace('.gif', '.jpg')

            source_sample_path = data_src_root + '/' + input_type + '/' + _class + '/' + file_name_no_ext + '.jpg'

            target_sample_path = data_dest_root+'/'+input_type+'/'+_class+'/'+file_name_no_ext+'.jpg'
            target_data_list.append([input_type,_class,file_name_no_ext,1])

            if not os.path.exists(os.path.dirname(target_sample_path)):
                os.makedirs(os.path.dirname(target_sample_path))

            im = cv2.imread(source_sample_path)
            im = cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(target_sample_path, im)

            pbar.update(1)


    write_data(data_dest_root,target_data_list)


def main():
    data_src_root = '/mnt/sda2/dev_root/dataset/combined/balanced-14-9/'
    data_dest_root = '/mnt/sda2/dev_root/dataset/combined/balanced-14-9-48x48/'

    generate(data_src_root,data_dest_root,48)

if __name__ == '__main__':
    main()
