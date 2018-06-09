from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self,pretrained_model,layer_name,size):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.size=size

        if pretrained_model is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        elif layer_name is not None:
            self.model = Model(
                inputs=pretrained_model.input,
                outputs=pretrained_model.get_layer(layer_name).output
            )

        else:
            # Load the model first.
            self.model = pretrained_model

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=self.size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        return features[0]
