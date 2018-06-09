import os
from random import shuffle
from shutil import copy

from tqdm import tqdm

from util.csv_util import get_data, write_data

NB_CLASSES=7

def generate(data_src_root,data_dest_root,one_class):
    """
    Downsample for 'one' class

    :param data_src_root:
    :param data_dest_root:
    :param one_class:
    :return:
    """
    data_file=data_src_root+'/data.csv'
    data_dest_root=data_dest_root+'/'+one_class+'_VS_rest'

    _source_data=get_data(data_file)
    shuffle(_source_data)

    train_data={}
    val_data={}

    for data in _source_data:
        type,clazz,_,_ = data

        if type =='Train' and not clazz in train_data:
            train_data[clazz]=[]

        if type == 'Val' and not clazz in val_data:
            val_data[clazz] = []

        if type == 'Train':
            train_data[clazz].append(data)

        if type == 'Val':
            val_data[clazz].append(data)



    CLASS_TRAIN_DATA_COUNT = len(train_data[one_class])
    CLASS_VAL_DATA_COUNT = len(val_data[one_class])

    one_class_train=train_data[one_class]
    one_class_val=val_data[one_class]

    rest_class_train=[]
    rest_class_val=[]
    nb_rest_classes=NB_CLASSES-1
    NB_TRAIN_SAMPLE_PER_REST_CLASS=CLASS_TRAIN_DATA_COUNT/nb_rest_classes
    NB_VAL_SAMPLE_PER_REST_CLASS=CLASS_VAL_DATA_COUNT/nb_rest_classes
    for clazz in train_data:
        clazz_list=train_data[clazz]
        rest_class_train.extend(clazz_list[0:min(len(clazz_list),NB_TRAIN_SAMPLE_PER_REST_CLASS)])
    for clazz in val_data:
        clazz_list = val_data[clazz]
        rest_class_val.extend(clazz_list[0:min(len(clazz_list),NB_VAL_SAMPLE_PER_REST_CLASS)])

    target_data_list=[]

    source_data=one_class_train+rest_class_train+one_class_val+rest_class_val

    with tqdm(desc='Generating oneVsrest data for class {}'.format(one_class), total=len(source_data), unit='it', unit_scale=False) as pbar:
        for sample in source_data:
            input_type, _class, file_name_no_ext, nb_subsample = sample

            nb_subsample = int(nb_subsample)

            if nb_subsample > 1:
                raise Exception('Video data not supported!')

            if nb_subsample == 0:
                raise Exception('No sample!')

            source_sample_path = data_src_root + '/' + input_type + '/' + _class + '/' + file_name_no_ext + '.jpg'

            if _class!=one_class:
                _class='Zrest' # prepend Z to ensure when ordered by name this is always second class


            target_sample_path = data_dest_root+'/'+input_type+'/'+_class+'/'+file_name_no_ext+'.jpg'
            target_data_list.append([input_type,_class,file_name_no_ext,1])

            if not os.path.exists(os.path.dirname(target_sample_path)):
                os.makedirs(os.path.dirname(target_sample_path))

            copy(source_sample_path,target_sample_path)

            pbar.update(1)


    write_data(data_dest_root,target_data_list)


def main():
    data_src_root = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN/'
    data_dest_root = '/mnt/sda2/dev_root/dataset/combined/FER13-SFEW-YALE-CKP-224-ONE-VS-REST-SUBSAMPLED-VAl-BALANCED'
    data_dest_root = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN-ONE-VS-REST-SUBSAMPLED-VAl-BALANCED/'

    generate(data_src_root,data_dest_root,'Angry')
    generate(data_src_root,data_dest_root,'Disgust')
    generate(data_src_root,data_dest_root,'Happy')
    generate(data_src_root,data_dest_root,'Neutral')
    generate(data_src_root,data_dest_root,'Sad')
    generate(data_src_root,data_dest_root,'Surprise')
    generate(data_src_root,data_dest_root,'Fear')

if __name__ == '__main__':
    main()
