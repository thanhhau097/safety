from utils import *
from tcn import compiled_tcn
from sklearn.model_selection import train_test_split
import pandas as pd


class Model():
    def __init__(self):
        self.model = compiled_tcn(return_sequences=False,
                                 num_feat=9,
                                 num_classes=2,
                                 nb_filters=64,
                                 kernel_size=9,
                                 dilations=[2**i for i in range(12)],
                                 nb_stacks=1,
                                 max_len=None,
                                 use_skip_connections=True)

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def load_weights(self, filepath):
        self.model.load_weights(filepath=filepath)
