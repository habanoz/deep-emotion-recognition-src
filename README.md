# deep-emotion-recognition-src

<h3>Description For Files</h3>

- train_image.py : This is the file for creating a model using transfer learning using one of previously trained networks.
- train_image_oneVres.py : This file is for generating expert models for each emotion.
- train_image_merge_models.py : This file is for stacking models using a two layer MLP.
- train_image_ptest_sample.py : This file is like train_image.py but for being run multiple times from run_train_image_ptest_sample.sh
- train_image_merge_ptest_sample.py : This file is like train_image_merge_models.py but for being run multiple times from run_train_image_merge_ptest_sample.sh

<h3>How to start</h3>

Extract the dataset archive. In train_image.py do following steps:
- Point DATA_DIR to location of dataset.
- Specify WEIGHTS_FILE to one of inception, resnet, vgg16 or vggface. Path to a hdf5 file which contains network weights with network structure can be specified, as well.
- Specify a WORK_DIR location of wish.
- And run. 
- Results will be written under WORK_DIR. Models will be saven under WORK_DIR/checkpoints directory. WORK_DIR/logs directory will cotain training history.
- Confustion matrices, validation results, accuracy by epoch chart will be generated.

