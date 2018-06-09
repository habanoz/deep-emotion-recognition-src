#!/usr/bin/env python

import glob
import os
import cv2
import numpy as np
from thirdp.c3d_keras import c3d_model
from keras.models import model_from_json


def main():
    root_dir='./c3d_keras/'
    model_dir = root_dir+'models'

    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")
    model.compile(loss='mean_squared_error', optimizer='sgd')

    print("[Info] Loading labels...")
    with open(root_dir+'sports1m/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    sub_samples = glob.glob('/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_PFaces_16_171_128/Train/Angry/000046280*')

    vid = []
    for sub_sample in sub_samples:
        vid.append(cv2.imread(sub_sample))

    vid = np.array(vid, dtype=np.float32)

    # sample 16-frame clip
    #start_frame = 100
    start_frame = 0
    X = vid[start_frame:(start_frame + 16), :, :, :]

    # subtract mean
    #mean_cube = np.load(root_dir+'models/train01_16_128_171_mean.npy')
    #mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
    #X -= mean_cube

    # center crop
    X = X[:, 8:120, 30:142, :] # (l, h, w, c)

    # input_shape = (16,112,112,3)

    # get activations for intermediate layers if needed
    inspect_layers = [
        'fc6',
        'fc7',
        ]
    for layer in inspect_layers:
        int_model = c3d_model.get_int_model(model=model, layer=layer)
        int_output = int_model.predict_on_batch(np.array([X]))
        int_output = int_output[0, ...]
        print "[Debug] at layer={}: output.shape={}".format(layer, int_output.shape)

    # inference
    output = model.predict_on_batch(np.array([X]))

    # show results
    print('Position of maximum probability: {}'.format(output[0].argmax()))
    print('Maximum probability: {:.5f}'.format(max(output[0])))
    print('Corresponding label: {}'.format(labels[output[0].argmax()]))

    # sort top five predictions from softmax output
    top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    print('\nTop 5 probabilities and labels:')
    for i in top_inds:
        print('{1}: {0:.5f}'.format(output[0][i], labels[i]))

if __name__ == '__main__':
    main()
