import os
from thirdp.harvitronix.extract.csv_file_constats import NB_SUB_INDEX, CLASS_INDEX, SAMPLE_INDEX
from util.generator.CvsGeneratorBase import GeneratorBase, IGeneratorBase


class MergeGenerator(GeneratorBase):
    def __init__(self, generators, csv_file_paths, is_train,batch_size):

        self.generators = generators
        self.is_train = is_train
        self.batch_size=batch_size

        data_sets = [self.get_data(file, is_train) for file in csv_file_paths]
        data_roots = [os.path.dirname(path) for path in csv_file_paths]
        self.data = self.filter_data_sets(data_sets)

        self.classes = self.get_classes(self.data)
        self.igenerators=[]

        # configure each generator
        for generator, data_root in zip(self.generators, data_roots):
            igenerator=generator.flow_from_csv_data(self.data, data_root,self.batch_size, is_train)
            self.igenerators.append(igenerator)

        super(MergeGenerator, self).__init__(0)

        self.flow_called=False

    def flow(self):
        self.flow_called = True
        return self.flow_from_csv_data(self.data, None,self.batch_size, self.is_train)

    def flow_from_csv_file(self, csv_file_path,batch_size, is_train):
        raise Exception("Not supported!")

    def flow_from_csv_data(self, data, data_root,batch_size, is_train):
        if not self.flow_called:
            raise Exception("Direct call not allowed. Call flow_from_csv_files method!")

        return self.get_generator(data, None, batch_size=batch_size, shuffle=is_train)

    def get_generator(self, data, data_root, batch_size, shuffle=False):
        return IMergeIGenerator(data, self.igenerators, batch_size, shuffle,classes=self.classes)

    def filter_data_sets(self, data_sets):
        """Compare data sets row by row. Each row in all files must be non zero. THis implementation assumes each sample
        happens to be at the same line number in all files. If a sample happens to be non zero in all files, sample row from first file is used.
        Thus sub sample count(nb_seq) information for all other files is lost.
        """

        res = []
        for same_sample_from_all_sources in zip(*data_sets):
            # array of sample names
            sample_names = [sample[SAMPLE_INDEX] for sample in same_sample_from_all_sources]
            # array filled with name of first sample
            base_sample_names = [same_sample_from_all_sources[0][SAMPLE_INDEX] for i in range(len(same_sample_from_all_sources))]

            # make sure that sample ordering is the same for all input files
            if sample_names!=base_sample_names:
                raise Exception("Sample names must be equal. Check sample ordering. Names:"+", ".join('{}'.format(x for x in sample_names)))

            non_zero_sample_count_flags = [int(sample[NB_SUB_INDEX]) > 0 for sample in same_sample_from_all_sources]
            if all(non_zero_sample_count_flags):
                res.append(same_sample_from_all_sources[0])

        return res

    def get_classes(self, data):
        """Extract the classes from our data."""
        classes = []
        for item in data:
            if item[CLASS_INDEX] not in classes:
                classes.append(item[CLASS_INDEX])

        # Sort them.
        classes = sorted(classes)

        # Return.
        return classes


class IMergeIGenerator(IGeneratorBase):
    def __init__(self, data, igenerators, batch_size, shuffle, seed=None, classes=None):
        self.igenerators = igenerators

        if not classes:
            raise Exception('Provide classes')

        super(IMergeIGenerator, self).__init__(data, None, batch_size, shuffle=shuffle, seed=seed, classes=classes, sample_suffix=None)

    def _get_batches_of_transformed_samples(self, index_array):
        input_set=[]
        output=None
        for generator in self.igenerators:
            X,y = generator._get_batches_of_transformed_samples(index_array)
            input_set.append(X)
            output=y

        return input_set, output
