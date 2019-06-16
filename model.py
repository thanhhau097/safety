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

    def load_weights(self, filepath):
        self.model.load_weights(filepath=filepath)

    def train(self, data_folder='data'):
        batch_size = 64
        X, y, _, _ = get_data(data_folder)

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05)

        train_gen = generator(X_train, y_train, batch_size)
        val_gen = generator(X_val, y_val, batch_size)

        self.model.fit_generator(train_gen, epochs=100, steps_per_epoch=len(y_train) // batch_size,
                                 validation_data=val_gen, validation_steps=len(y_val) // batch_size)

    def test(self, folder):
        batch_size = 1
        X, y, bookingIDs, ignored_bookingIDs = get_data(folder)

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.load_weights('tcn.h5')
        print("LOAD WEIGHTS COMPLETELY")

        n_cases = 10
        test_gen = generator_test(X, y, batch_size, n_cases)
        y_pred = self.model.predict_generator(test_gen, steps=len(y) // batch_size, verbose=1)

        y_pred = np.argmax(y_pred, axis=1).reshape([-1, n_cases])
        y_pred = np.sum(y_pred, axis=1)
        y_pred[y_pred <= 5] = 0
        y_pred[y_pred > 5] = 1

        acc = np.sum(np.array(y) == np.array(y_pred)) / len(y)
        print("Accuracy =", acc)

        print("WRITING RESULT TO FILE")
        bookingIDs += list(ignored_bookingIDs)
        y_pred = list(y_pred) + [0] * len(ignored_bookingIDs)
        self.write_test_result('result.csv', bookingIDs, y_pred)

    def write_test_result(self, filepath, bookingIDs, y):
        df = pd.DataFrame()
        df['bookingID'] = bookingIDs
        df['label'] = y

        df.to_csv(filepath)