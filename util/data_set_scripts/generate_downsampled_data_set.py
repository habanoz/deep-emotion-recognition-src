import os
from random import shuffle, random
from shutil import copy
from tqdm import tqdm
from util.csv_util import get_data, write_data

NB_CLASSES=7

def generate(data_src_root,data_dest_root,class_sample_rates):
    """
    Downsample for 'one' class

    :param data_src_root:
    :param data_dest_root:
    :param class_sample_rates: map. class names are keys. Values are list of probabilities for selecting an instance.
    [train_sample_selection_probability,validation_sample_selection_probability]
    :return:
    """
    data_file=data_src_root+'/data.csv'

    source_data=get_data(data_file)
    shuffle(source_data)

    target_data_list=[]

    with tqdm(desc='Generating downsampled data', total=len(source_data), unit='it', unit_scale=False) as pbar:
        for sample in source_data:
            input_type, _class, file_name_no_ext, nb_subsample = sample

            probabilities_per_type = class_sample_rates[_class]
            probability = probabilities_per_type[0] if input_type=='Train' else probabilities_per_type[1]

            probability = 1-probability

            if probability>random():
                pbar.update(1)
                continue

            nb_subsample = int(nb_subsample)

            if nb_subsample > 1:
                raise Exception('Video data not supported!')

            if nb_subsample == 0:
                raise Exception('No sample!')

            source_sample_path = data_src_root + '/' + input_type + '/' + _class + '/' + file_name_no_ext + '.jpg'

            target_sample_path = data_dest_root+'/'+input_type+'/'+_class+'/'+file_name_no_ext+'.jpg'
            target_data_list.append([input_type,_class,file_name_no_ext,1])

            if not os.path.exists(os.path.dirname(target_sample_path)):
                os.makedirs(os.path.dirname(target_sample_path))

            copy(source_sample_path,target_sample_path)

            pbar.update(1)

    write_sample_count(data_dest_root,target_data_list)
    write_data(data_dest_root,target_data_list)

def write_sample_count(data_dest_root,target_data_list):
    train_data={}
    val_data={}

    for sample in target_data_list:
        type, clazz, file_name_no_ext, nb_subsample = sample

        if type == 'Train' and not clazz in train_data:
            train_data[clazz] = 0

        if type == 'Val' and not clazz in val_data:
            val_data[clazz] = 0

        if type == 'Train':
            train_data[clazz]+=1

        if type == 'Val':
            val_data[clazz]+=1

    with open(data_dest_root+'/sample_counts.txt','w') as f:
        f.write("Train samples:\n")
        for key, value in train_data.iteritems():
            f.write(key+' : '+str(value)+'\n')

        f.write("\n")

        f.write("Validation samples:\n")
        for key, value in val_data.iteritems():
            f.write(key+' : '+str(value)+'\n')


def main():
    data_src_root = '/mnt/sda2/dev_root/dataset/original/fer2013_224_NEW_CLEAN'
    data_dest_root = '/mnt/sda2/dev_root/dataset/downsampled/fer2013_224_high_accuracy_less_frequent_2'

    generate(data_src_root,data_dest_root,
             {'Angry':[1.0,1.0],'Disgust':[1.0,1.0],'Fear':[1.0,1.0],'Happy':[0.25,1.0],'Neutral':[1.0,1.0],'Sad':[1.0,1.0],'Surprise':[0.5,1.0]}
             )

if __name__ == '__main__':
    main()
