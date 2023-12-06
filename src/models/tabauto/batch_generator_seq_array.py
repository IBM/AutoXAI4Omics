import numpy as np
from tensorflow.keras.utils import Sequence

# from tensorflow.keras.utils import to_categorical

import logging

_log = logging.getLogger(__name__)


class BatchGeneratorSeqArray(Sequence):
    def __init__(
        self,
        dataset_x,
        dataset_y,
        dataset_labels=[],
        batch_size=32,
        shuffle=False,
        seed=None,
        aug=None,
        one_dim=False,
        transform_label=True,
        preprocessing_function=None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.one_dim = one_dim
        self.transform_label = transform_label
        self.preprocessing_function = preprocessing_function

        self.rs = np.random.RandomState(seed)

        assert self.batch_size > 0, "Batch size has to be a positive integer!"

        # generate pointers to the data
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

        self.label_table = []
        if dataset_labels is None:
            # build a list of labels based on the number of classes (from self.datataset_y)
            self.label_table = np.unique(self.dataset_y)
            pass
        else:
            for element in dataset_labels:
                # self.label_table.append(element.decode('UTF-8'))
                self.label_table.append(element)

        self.preprocessing_config = None

        assert self.dataset_x.shape[0] == self.dataset_y.shape[0]

        # the ordering in the container

        # Preload all the labels.
        self.labels = self.dataset_y[:]

        self.len = self.dataset_x.shape[0]
        self.indexes = np.arange(self.len)
        if self.shuffle is True:
            self.rs.shuffle(self.indexes)

    # ACCESS METADATA
    def get_preprocessing_config(self):
        return self.preprocessing_config

    def get_label_table(self):
        return self.label_table

    def get_num_classes(self):
        return len(self.label_table)

    # ACCESS DATA AND SHAPES #
    def get_num_samples(self):
        return self.dataset_x.shape[0]

    def __len__(self):
        self.len = self.dataset_x.shape[0]
        return int(np.ceil(self.len / float(self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        # check if end_idx > len
        if end_idx > self.len:
            end_idx = self.len

        access_pattern = self.indexes[start_idx:end_idx]
        access_pattern = sorted(access_pattern)

        x_train = self.dataset_x[access_pattern, :]

        if self.preprocessing_function is not None:
            arrays = []
            for k in range(0, end_idx - start_idx):
                im = self.preprocessing_function(x_train[k, ...])
                arrays.append(im)
            x_train = np.stack(arrays)

        """
        y_train = self.labels[access_pattern]

        num_classes = self.get_num_classes()
        if self.transform_label:
            y_train = to_categorical(y_train, num_classes)
        """
        y_train = self.dataset_y[access_pattern, :]

        return x_train, y_train

    def on_epoch_end(self):
        if self.shuffle is True:
            self.rs.shuffle(self.indexes)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
