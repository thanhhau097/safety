from sklearn.model_selection import train_test_split

from model import Model
from utils import *


def train(data_folder='data'):
    model = Model()
    batch_size = 64
    X, y, _, _ = get_data(data_folder)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05)

    train_gen = generator(X_train, y_train, batch_size)
    val_gen = generator(X_val, y_val, batch_size)

    model.model.fit_generator(train_gen, epochs=100, steps_per_epoch=len(y_train) // batch_size,
                             validation_data=val_gen, validation_steps=len(y_val) // batch_size)

if __name__ == "__main__":
    train(data_folder='data')




