import os

import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.preprocessing import sequence
import numpy as np


def read_data(folder):
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
    # có normalize theo cột hay không?
    # càn kiểm tra những timestep cuối có Speed = 0 hay không, nếu có thì bỏ đi để giảm kích thước dữ liệu, vì nếu dừng lại rồi thì không ảnh hưởng
    # đến kết quả là nguy hiểm hay không.
    # kiểm tra để loại bỏ tốc độ xuất phát = 0 và tốc độ kết thúc = 0
    # lưu ý: trường second có những trường hợp không phải liên tục 0, 1, 2... cần xử lý những second cách nhau quá xa?
    # các bước thực hiện: -> groupby booking -> loại theo speed -> normalize -> groupby booking -> get values
    features = features[features.Accuracy <= 100]

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
    features, labels = read_data(folder)
    X, y, bookingIDs, ignored_bookingIDs = preprocess_data(features, labels)

    return X, y, bookingIDs


def ignore_velocity_at_post(X, y, bookingIDs):
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
    new_X = []

    for j, x in enumerate(X):
        X_i = []
        for i in range(len(x) // 2):
            X_i.append(x[i * 2 + np.random.randint(2)])

        new_X.append(X_i)
    return new_X, y


def generator(X, y, batch_size=32):
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
