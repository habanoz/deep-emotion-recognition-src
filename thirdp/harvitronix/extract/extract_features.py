"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import os.path

import numpy as np
from harvitronix.extract.data import DataSet
from harvitronix.extract.data import FILE_INDEX
from tqdm import tqdm

from extractor import Extractor

# Set defaults.
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.


def extractor_features(data_file,sequences_dir,seq_length,pretrained_model=None,layer_name=None,size=(150,150)):

    if not os.path.exists(sequences_dir):
        os.makedirs(sequences_dir)

    # Get the dataset.
    data = DataSet(data_file,sequences_dir,seq_length=seq_length, class_limit=class_limit)
    # get the model.
    model = Extractor(pretrained_model,layer_name,size)
    # Loop through data.
    pbar = tqdm(total=len(data.data))
    for video in data.data:

        # Get the path to the sequence for this video.
        path = sequences_dir + '/' + video[FILE_INDEX] + '-' + str(seq_length) + '-features.txt'

        # Check if we already have it.
        if os.path.isfile(path):
            pbar.update(1)
            continue

        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)

        # Now downsample to just the ones we need.
        frames = data.rescale_list(frames, seq_length)

        # Now loop through and extract features to build the sequence.
        sequence = []
        for image in frames:
            features = model.extract(image)
            sequence.append(features)

        # Save the sequence.
        np.savetxt(path, np.array(sequence).reshape((seq_length,-1)))

        pbar.update(1)
    pbar.close()

def main():
    data_file = './data/data_file.csv'
    sequences_dir = './data/sequences/'
    extractor_features(data_file,sequences_dir,)

if __name__ == '__main__':
    main()
