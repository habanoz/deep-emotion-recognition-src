import os

import numpy
from tensorflow.python.framework.ops import get_default_graph

from thirdp.harvitronix.extract.csv_file_constats import NB_SUB_INDEX, CLASS_INDEX, SAMPLE_INDEX
from util.generator.CvsGeneratorBase import GeneratorBase, IGeneratorBase
from util.generator.MergeCsvGenerator import IMergeIGenerator, MergeGenerator


class InFlightMergeGenerator(MergeGenerator):
    def __init__(self,models, generators, csv_file_paths, is_train,batch_size):

        self.models=models

        super(InFlightMergeGenerator, self).__init__(generators, csv_file_paths, is_train,batch_size)

    def get_generator(self, data, data_root, batch_size, shuffle=False):
        return InFlightIMergeIGenerator(self.models, data, self.igenerators, batch_size, shuffle,classes=self.classes)



class InFlightIMergeIGenerator(IMergeIGenerator):
    def __init__(self, models, data, igenerators, batch_size, shuffle, seed=None, classes=None):
        self.models=models
        self.graph = get_default_graph()
        super(InFlightIMergeIGenerator, self).__init__(data, igenerators, batch_size, shuffle=shuffle, seed=seed, classes=classes)

    def _get_batches_of_transformed_samples(self, index_array):

        # super generator returns batch of input data
        X, y = super(InFlightIMergeIGenerator, self)._get_batches_of_transformed_samples(index_array)

        Xp=[]
        for Xi,model in zip(X,self.models):
            with self.graph.as_default():
                # use model and batch of input data to create batch of predictions, in other words batch of
                Xpi=model.predict_on_batch(Xi)
                Xp.append(Xpi)

        return numpy.concatenate(Xp,axis=1), y
