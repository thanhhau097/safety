import numpy as np

from model import Model
from utils import *


def write_test_result(filepath, bookingIDs, y):
    df = pd.DataFrame()
    df['bookingID'] = bookingIDs
    df['label'] = y

    df.to_csv(filepath)


def test(folder):
    model = Model()

    batch_size = 1
    X, y, bookingIDs, ignored_bookingIDs = get_data(folder)

    model.load_weights('tcn.h5')
    print("LOAD WEIGHTS COMPLETELY")

    n_cases = 10
    test_gen = generator_test(X, y, batch_size, n_cases)
    y_pred = model.model.predict_generator(test_gen, steps=len(y) // batch_size, verbose=1)

    y_pred = np.argmax(y_pred, axis=1).reshape([-1, n_cases])
    y_pred = np.sum(y_pred, axis=1)
    y_pred[y_pred <= n_cases // 2] = 0
    y_pred[y_pred > n_cases // 2] = 1

    acc = np.sum(np.array(y) == np.array(y_pred)) / len(y)
    print("Accuracy =", acc)

    print("WRITING RESULT TO FILE")
    bookingIDs += list(ignored_bookingIDs)
    y_pred = list(y_pred) + [0] * len(ignored_bookingIDs)
    write_test_result('result.csv', bookingIDs, y_pred)


if __name__ == "__main__":
    test(folder='data')