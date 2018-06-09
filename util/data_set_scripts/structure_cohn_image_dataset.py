import os
import csv
import glob
import shutil
from PIL import Image

from util.directory_utils import get_path_parts




def structure(source_root, target_root):
    """
    Cohn Canade face data set is inform of
    /mnt/sda2/dev_root/dataset/original/cohn-kanade-images/Emotion/subject-i/exp-j/subject-i_exp-j_000000N_emotion.txt

    subject-i can be like S005
    exp-j can be like 001

    cohn-kanade-images/subject-i/exp-j/subject-i_exp-j_00000001.png
    cohn-kanade-images/subject-i/exp-j/subject-i_exp-j_00000002.png
    ...
    cohn-kanade-images/subject-i/exp-j/subject-i_exp-j_0000000N.png


    Use only records with emotion label. Data set contains sequences, only the last one expresses full emotion.
    Label files only gives emotion for last image of an expression sequence. First images are taken as neutral images but same subject
    should only appear once in neutral. While saving neutral images expression part changed as 0 to overwrite any existing
    neutral images of the same subject.

    0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

    Contempt is not included.

    Moved to correct input folder e.g. Train, Val, class folder and data file is created.

    train subjects are upto subject 116.

    :return:
    """

    # index is compliant with emotion index found in label files, see above
    target_classes=['Neutral','Angry','Disgust','Fear','Happy','Sad','Surprise']

    if not os.path.exists(target_root):
        os.makedirs(target_root)

    label_files=glob.glob(source_root + '/Emotion/*/*/*.txt')


    last_subject_index = 115
    data_file = []

    nb_train=0
    nb_val=0
    neutral_dict={}
    for label_file in label_files:
        parts  = label_file.split('/')
        subject_dir, expression_dir,label_file_name=parts[-3],parts[-2],parts[-1] # last part is label file name, grap previous parts
        subject_idx=int(subject_dir[1:]) # exclude S prefix and convert to int

        input_folder='Train' if subject_idx<=last_subject_index else 'Val'

        with open(label_file, 'r') as f:
            class_idx = int(float(f.readline().strip()))

        if class_idx==2: # contempt is not used
            print 'Contempt detected, ignoring'
            continue

        class_folder=target_classes[class_idx]
        image_file_no_ext=label_file_name.replace('_emotion.txt', '')

        source_sample_dir=source_root+'/'+subject_dir+'/'+expression_dir+'/'
        source_sample_path=source_sample_dir+image_file_no_ext+'.png'
        target_sample_input_dir=target_root+'/'+input_folder+'/'
        target_sample_class_dir=target_sample_input_dir+class_folder+'/'
        target_sample_neutral_dir=target_sample_input_dir+target_classes[0]+'/'
        target_sample_path=target_sample_class_dir+image_file_no_ext+'.jpg'

        if not os.path.exists(target_sample_class_dir):
            os.makedirs(target_sample_class_dir)

        if not os.path.exists(target_sample_neutral_dir):
            os.makedirs(target_sample_neutral_dir)

        img = Image.open(source_sample_path)
        img.save(target_sample_path)

        data_file.append([input_folder, class_folder, image_file_no_ext, 1])

        print('File {} processed'.format(source_sample_path))

        nb_train = nb_train+1 if input_folder=='Train' else nb_train
        nb_val = nb_val+1 if input_folder=='Val' else nb_val

        # neutral should be added for each subject once
        if subject_dir in neutral_dict:
            continue

        # now copy first/neutral file
        source_sample_path = source_sample_dir + subject_dir+'_'+expression_dir +'_00000001'+ '.png'

        target_neutral_file_name_without_ext=subject_dir +'_'+str(0) +'_00000001'
        target_sample_path = target_sample_neutral_dir + target_neutral_file_name_without_ext + '.jpg'

        img = Image.open(source_sample_path)
        img.save(target_sample_path)

        neutral_dict[subject_dir]=subject_dir
        class_folder=target_classes[0]
        data_file.append([input_folder, class_folder, target_neutral_file_name_without_ext, 1])

        nb_train = nb_train + 1 if input_folder == 'Train' else nb_train
        nb_val = nb_val + 1 if input_folder == 'Val' else nb_val

    print("Train samples {} test samples {}".format(nb_train,nb_val))

    with open(target_root+'/'+'data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)




def main():
    data_src_root = '/mnt/sda2/dev_root/dataset/original/cohn-kanade-images'
    data_dest_root = '/mnt/sda2/dev_root/dataset/original/cohn-kanade-images-structured'

    structure(data_src_root,data_dest_root)

if __name__ == '__main__':
    main()
