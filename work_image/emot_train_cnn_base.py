from work_image.keras_train_cnn_base import KerasTrainCnnBase

NB_CLASSES = 7

class EmotTrainCnnBase(KerasTrainCnnBase):
    def __init__(self, work_dir, config ):

        super(KerasTrainCnnBase, self).__init__(work_dir, config=config,nb_classes=NB_CLASSES)

        return