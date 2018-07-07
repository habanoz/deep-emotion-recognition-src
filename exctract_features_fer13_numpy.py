import csv
import os
import pickle
from keras.engine.training import Model
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

"""
 load model, extract fer13 features and save as pickle object
"""

def get_data(data_file):
    """Load our data from file."""
    with open(data_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data


emot_model = "D:\dev_root\work2.1\merged\image-whole\\1__1512579280.12-17403535-1-vggface-aligned-catvrossloss\checkpoints\w.007-0.6779-2.79.hdf5"
data_dir = "D:\dev_root\dataset\original\\fer2013_224_NEW_CLEAN"
target_data_dir = data_dir + "\Val"
layer_name="fc6_1"

sample_files = [r+'/'+file for r, d, files in os.walk(target_data_dir) for file in files]

num_samples=len(sample_files)

model = load_model(emot_model)

model.summary()

model = Model(
    inputs=model.input,
    outputs=model.get_layer(layer_name).output
)

features_dict = {}
features=None

for sample_file in sample_files:
    sample_file_base=os.path.basename(sample_file)

    img = load_img(sample_file,target_size=(224,224))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    features = model.predict_on_batch(x)
    features_dict[sample_file_base]=features[0]

num_features=len(features)

with open("val_emot_"+layer_name+"_"+str(num_features)+"_features" + '.pkl', 'wb') as f:
    pickle.dump(features_dict, f, pickle.HIGHEST_PROTOCOL)