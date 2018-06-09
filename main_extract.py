from extract.adjust_subsample_count import AdjustSubsampleCount
from extract.draw_landmarks import DrawLandmarks
from extract.exctract_faces import ExtractFaces

import os

from extract.exctract_landmarks import ExtractLandmarks
from extract.exctract_landmarks_fix_seq_length import ExtractLandmarksFixLength
from extract.preprocess.preprocess_images import PreprocessFaces


# This class is for extracting various feature from videos/images
#
# Use extractors in correct order and generate required features
#

def main():

    dimension=(224, 224)
    nb_sequences = 71

    # 1 extract frames from videos parameters
    video_to_frames_source_dir = '/mnt/sda2/dev_root/dataset/AFEW_6_2016'
    frames_dir = '/mnt/sda2/dev_root/dataset/original/google_extracted_emotions/'
    frames_dir = '/mnt/sda2/dev_root/dataset/original/cohn-kanade-images-structured/'
    frames_dir = '/mnt/sda2/dev_root/dataset/original/sfew/'
    frames_dir = '/mnt/sda2/dev_root/dataset/original/yalefaces-structured/'
    adjusted_frames_dir = '/mnt/sda2/dev_root/dataset/CK+/cohn-kanade-frames-224-71/'
    landmarks_dir = '/mnt/sda2/dev_root/dataset/CK+/cohn-kanade-lm-224-71fx/'
    landmarks_draw_dir = '/mnt/sda2/dev_root/dataset/CK+/cohn-kanade-lm-draw-224-71/'

    # 2 extract frames from videos parameters
    faces_dir = '/mnt/sda2/dev_root/dataset/google_extracted/emotions_faces_aligned_p01/'
    faces_dir = '/mnt/sda2/dev_root/dataset/CK+/CKP_Faces_224_p01/'
    faces_dir = '/mnt/sda2/dev_root/dataset/sfew/Aligned_Faces_224_p01/'
    faces_dir = '/mnt/sda2/dev_root/dataset/yale-Processed/not_aligned_yale_Faces_224/'

    # 3 preprocess images
    pfaces_dir = '/mnt/sda2/dev_root/dataset/google_extracted/emotions_pfaces_aligned_p01/'
    pfaces_dir = '/mnt/sda2/dev_root/dataset/CK+/CKP_PFaces_224_p01/'
    pfaces_dir = '/mnt/sda2/dev_root/dataset/sfew/Aligned_PFaces_224_p01/'
    pfaces_dir = '/mnt/sda2/dev_root/dataset/yale-Processed/not_aligned_yale_PFaces_224/'

    # 4 extract features
    features_dir = '/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Features_fc7_vgg16face_120_'+str(nb_sequences)


    # fixed subsamples
    new_subsample_dir='/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_PFaces_2_112/'

    # inception
    #model_file='/mnt/sda2/dev_root/work/inception/inception_1/checkpoints/m.hdf5'
    #weights_file='/mnt/sda2/dev_root/work/inception/inception_1/checkpoints/w.030-0.57-2.03.hdf5'
    #layer_name = 'global_average_pooling2d_1'

    # vggface
    #model_file = '/mnt/sda2/dev_root/work/vgg16face/vggface_1_120/checkpoints/m.hdf5'
    #weights_file = '/mnt/sda2/dev_root/work/vgg16face/vggface_1_120/checkpoints/w.009-0.66-2.84.hdf5'
    #layer_name = 'fc7'

    # vgg16
    #model_file='/mnt/sda2/dev_root/work/vgg16/fer13/vgg16_12_120/checkpoints/m.hdf5'
    #weights_file='/mnt/sda2/dev_root/work/vgg16/fer13/vgg16_12_120/checkpoints/w.028-0.67-2.24.hdf5'
    #layer_name = 'dense_1'

    # vgg16 imagenet
    #model = VGG16(weights="imagenet",include_top=True)
    #layer_name = 'flatten'

    #model=load_model(model_file)
    #model.load_weights(weights_file)


    #for i, layer in enlbcumerate(model.layers):
    #    print(i, layer.name)
    #    print(layer.get_output_at(0).get_shape().as_list())


    #model = None

    # 1 extract frames from videos
    #video_to_frames_processor = ProcessVideos(video_to_frames_source_dir, video_to_frames_target_dir)
    # video_to_frames_processor.process()

    # 2 extract faces from frames
    frames_to_faces_processor = ExtractFaces(frames_dir, faces_dir,skip_existing=True,align_faces=False)
    frames_to_faces_processor.process()


    # 3 preprocess faces
    faces_to_pfaces_processor = PreprocessFaces(faces_dir, pfaces_dir, dimension=(224,224),generate_data_file_only=False)
    faces_to_pfaces_processor.process()

    # 4 extract features
    #faces_to_pfaces_processor = ExtractFeatures(pfaces_dir, features_dir, seq_length=nb_sequences, pretrained_model=model, layer_name=layer_name, generate_data_file_only=False,dimension=dimension)
    #faces_to_pfaces_processor.process()


    # fix subsample count - if necessary
    #fix_subsample_processor=AdjustSubsampleCount(frames_dir, adjusted_frames_dir, dimension=dimension, seq_length=nb_sequences,use_padding=True,nb_min_subsample=10)
    #fix_subsample_processor.process()

    #lm_extractor = ExtractLandmarks(adjusted_frames_dir,landmarks_dir,seq_length=nb_sequences,generate_data_file_only=False,use_padding=True,nb_min_subsample=10)
    #lm_extractor.process()

    #lm_draw = DrawLandmarks(adjusted_frames_dir,landmarks_draw_dir,seq_length=nb_sequences,generate_data_file_only=False)
    #lm_draw.process()

    #lm_extractor = ExtractLandmarksFixLength(landmarks_dir, fixed_landmarks_dir, seq_length=nb_sequences, generate_data_file_only=False)
    #lm_extractor.process()

if __name__ == '__main__':
    main()

    os.system("paplay /usr/share/sounds/ubuntu/ringtones/Ubuntu.ogg")