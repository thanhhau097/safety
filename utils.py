import os

import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.preprocessing import sequence
import numpy as np


def read_data(folder):
    """
    Read features and labels from data folder
    :param folder: path to data folder
    :return:
    """
    i = 0

    features_path = os.path.join(folder, 'features')
    for file in tqdm(os.listdir(features_path)):
        if "csv" not in file:
            continue
        new_df = pd.read_csv(os.path.join(features_path, file))
        if i == 0:
            features = new_df
        else:
            features = pd.concat([features, new_df])
        i += 1

    labels_path = os.listdir(os.path.join(folder, 'labels'))[0]
    labels = pd.read_csv(os.path.join(folder, 'labels', labels_path))

    return features, labels


def preprocess_data(features, labels):
    """
    Preprocessing data

    :param features: dataframe from features folder
    :param labels: labels from labels folder
    :return:
        X: list of sequence features
        y: labels
        bookingIDs: list of bookingIDs
        ignored_bookingIDs: list of bookingIDs that are not used after preprocessing data
    """

    # ignore large error in GPS
    features = features[features.Accuracy <= 100]

    # feature scaling
    x = features.values[:, 1:]  # returns a numpy array
    min_max_scaler = preprocessing.MaxAbsScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_df = pd.DataFrame(x_scaled, columns=list(features.columns)[1:])
    new_df["bookingID"] = features.bookingID.values
    features = new_df

    X, y, bookingIDs = [], [], []
    for bookingID, label in tqdm(zip(labels.bookingID.values, labels.label.values)):
        book_df = features[features.bookingID == bookingID]
        book_df = book_df.sort_values(by=["second"])
        # ignore id column
        book_df = book_df.drop(columns=['second'])
        X_i = book_df.values[:, :-1]  # because bookingID is now last column, and ignore second
        X.append(X_i)
        y.append(label)
        bookingIDs.append(bookingID)

    X, y, bookingIDs, ignored_bookingIDs = ignore_velocity_at_post(X, y, bookingIDs)

    return X, y, bookingIDs, ignored_bookingIDs


def get_data(folder):
    """
    Get data for training or testing

    :param folder: path to data folder
    :return:
        X: list of sequence features
        y: labels
        bookingIDs: list of bookingIDs
        ignored_bookingIDs: list of bookingIDs that are not used after preprocessing data
    """
    features, labels = read_data(folder)
    X, y, bookingIDs, ignored_bookingIDs = preprocess_data(features, labels)

    return X, y, bookingIDs, ignored_bookingIDs


def ignore_velocity_at_post(X, y, bookingIDs):
    """
    Ignore the last time steps that those velocity are equal to 0
    :param X: list of sequence features
    :param y: labels
    :param bookingIDs: list of bookingIDs
    :return:
        X: list of sequence features
        y: labels
        bookingIDs: list of bookingIDs
        ignored_bookingIDs: list of bookingIDs that are not used after preprocessing data
    """
    new_X = []

    set_i = set()
    for k, x in tqdm(enumerate(X)):
        for i, element in enumerate(x[::-1]):
            if i == len(x) - 1:
                new_X.append(x)
                set_i.add(k)
            elif element[-1] != 0:
                new_X.append(x[:len(x) - i])
                set_i.add(k)
                break

    total_set = set(range(len(X)))

    diff = list(total_set.difference(set_i))[::-1]
    ignored_bookingIDs = []

    diff.sort()
    for i in diff[::-1]:
        ignored_bookingIDs.append(bookingIDs[i])
        del y[i]
        del bookingIDs[i]

    return new_X, y, bookingIDs, ignored_bookingIDs


def get_shorter_sequence(X, y):
    """
    Generating more data for training and testing

    :param X: list of sequence features
    :param y: labels
    :return:

    """
    new_X = []

    for j, x in enumerate(X):
        X_i = []
        for i in range(len(x) // 2):
            X_i.append(x[i * 2 + np.random.randint(2)])

        new_X.append(X_i)
    return new_X, y


def generator(X, y, batch_size=32):
    """
    Generator for training
    :param X: list of sequence features
    :param y: labels
    :param batch_size: batch size
    :return:
    """
    while True:
        X_temp, y_temp = shuffle(X, y)
        for i in range(len(y_temp) // batch_size):
            max_timesteps = 0
            for j in range(i * batch_size, (i + 1) * batch_size):
                timesteps = X_temp[j].shape[0]
                if max_timesteps < timesteps:
                    max_timesteps = timesteps

            x_batch = X_temp[i * batch_size: (i + 1) * batch_size]
            y_batch = y_temp[i * batch_size: (i + 1) * batch_size]

            x_batch, y_batch = get_shorter_sequence(x_batch, np.array(y_batch))

            x_batch = sequence.pad_sequences(x_batch, maxlen=max_timesteps, padding='post', dtype=np.float32)

            yield x_batch, np.array(y_batch)


def generator_test(X, y, batch_size=4, n_cases=10):
    """
    Generator for testing
    :param X: list of sequence features
    :param y: labels
    :param batch_size: batch size
    :param n_cases: numbers of cases that we use for each booking in test data
    :return:
    """
    while True:
        for i in range(len(y) // batch_size):
            max_timesteps = 0
            for j in range(i * batch_size, (i + 1) * batch_size):
                timesteps = X[j].shape[0]
                if max_timesteps < timesteps:
                    max_timesteps = timesteps

            x_batch = []
            for k in range(i * batch_size, (i + 1) * batch_size):
                for _ in range(n_cases):
                    x_batch.append(X[k])

            y_batch = y[i * batch_size: (i + 1) * batch_size]

            x_batch, y_batch = get_shorter_sequence(x_batch, np.array(y_batch))
            x_batch = sequence.pad_sequences(x_batch, maxlen=max_timesteps, padding='post', dtype=np.float32)

            yield x_batch, np.array(y_batch)
