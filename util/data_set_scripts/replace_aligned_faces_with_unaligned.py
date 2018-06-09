import os
import cv2
from tqdm import tqdm

from util.csv_util import get_data,write_data


def generate(data_src_root,data_dest_root,unaligned_face_dirs,dimension):
    data_file=data_src_root+'/data.csv'

    if dimension:
        size=(dimension,dimension)
    else:
        size=None

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

            source_sample_path=None

            for unaligned_face_root in unaligned_face_dirs:
                unaligned_face_path = unaligned_face_root + '/' + input_type + '/' + _class + '/' + file_name_no_ext + '.jpg'
                if os.path.exists(unaligned_face_path):
                    source_sample_path=unaligned_face_path

            if not source_sample_path:
                raise Exception("Sample unaligned face not found {}".format(input_type + '/' + _class + '/' + file_name_no_ext + '.jpg'))

            target_sample_path = data_dest_root+'/'+input_type+'/'+_class+'/'+file_name_no_ext+'.jpg'
            target_data_list.append([input_type,_class,file_name_no_ext,1])

            if not os.path.exists(os.path.dirname(target_sample_path)):
                os.makedirs(os.path.dirname(target_sample_path))

            im = cv2.imread(source_sample_path)
            if size:
                im = cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(target_sample_path, im)

            pbar.update(1)


    write_data(data_dest_root,target_data_list)


def main():
    data_src_root = '/mnt/sda2/dev_root/dataset/combined/balanced-14-9/'
    data_dest_root = '/mnt/sda2/dev_root/dataset/combined/balanced-unaligned-14-9-48x48/'
    unaligned_face_dirs=['/mnt/sda2/dev_root/dataset/FER13-Processed/FER13-224',
                         '/mnt/sda2/dev_root/dataset/sfew/pfaces-not-aligned-48',
                         '/mnt/sda2/dev_root/dataset/CK+/pfaces-not-aligned-48']
    generate(data_src_root,data_dest_root,unaligned_face_dirs,48)

if __name__ == '__main__':
    main()
