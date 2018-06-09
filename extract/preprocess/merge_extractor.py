from keras.legacy.layers import Merge
from keras.models import load_model, Sequential


class MergerExtractor():
    """
    Given multiple models, concatenates model outputs to generate singe merged output.

    Input must be multi-part to support all models. Each part input must be suitable for target model.

    """

    def __init__(self,model_files,nb_classes=7):

        models = []
        for model_file in model_files:
            model = load_model(model_file)
            models.append(model)

        merged_model = self.merge_models(models)

        self.merge_model=merged_model
        return

    def extract(self, input):

        features=self.merge_model.predict(input)

        return features[0]

    def merge_models(self, models):

        for i, model in enumerate(models):
            model.get_layer(name='flatten_1').name = 'flatten_1_' + str(i)
            model.get_layer(name='dense_1').name = 'dense_1_' + str(i)
            model.get_layer(name='dense_2').name = 'dense_2_' + str(i)
            model.get_layer(name='dropout_1').name = 'dropout_1_' + str(i)

            for layer in model.layers:
                layer.trainable = False

            model.summary()

        model = Sequential()
        model.add(Merge(models, mode='concat'))

        return model