import numpy as np
import matplotlib.pylab as plt


def load_data_helper(x_train, y_train, x_val, y_val, x_test, y_test):
    training_inputs = [np.reshape(x, (784, 1)) for x in x_train]
    training_results = [one_hot_encoding(y) for y in y_train]
    training_data = [tuple(item) for item in zip(training_inputs, training_results)]
    validation_inputs = [np.reshape(x, (784, 1)) for x in x_val]
    validation_data = [tuple(item) for item in zip(validation_inputs, y_val)]
    test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
    test_data = [tuple(item) for item in zip(test_inputs, y_test)]
    return (training_data, validation_data, test_data)


def one_hot_encoding(j, max=10):
    e = np.zeros((max, 1))
    e[j] = 1.0
    return e


def visualize_training_data(xs, ys, output):
    true_x = xs[output]
    true_y = ys[output]

    red_x = xs[np.logical_not(output)]
    red_y = ys[np.logical_not(output)]

    plt.title("Training Data Visualization")
    plt.scatter(true_x, true_y, s=0.25)
    plt.scatter(red_x, red_y, s=0.25)
    plt.xlabel("x_0")
    plt.ylabel("x_1")
    plt.axis("equal")
    plt.show()


def test_function_1(x, y):
    return x**2 + y > 0.4


def generate_test_train_val_data(split=0.70, visualize=True):
    (X, Y) = np.meshgrid(np.random.uniform(-1, 1, 300), np.random.uniform(-1, 1, 300))
    xs, ys = X.flatten(), Y.flatten()
    np.random.shuffle(xs)
    np.random.shuffle(ys)
    length = len(xs)

    train = [
        (np.array([[x, y]]).T, one_hot_encoding(test_function_1(x, y), max=2))
        for x, y in zip(xs[: int(length * split)], ys[: int(length * split)])
    ]
    test = [
        (np.array([[x, y]]).T, test_function_1(x, y))
        for x, y in zip(xs[int(length * split) :], ys[int(length * split) :])
    ]

    if visualize:
        visualize_training_data(xs, ys, [test_function_1(x, y) for x, y in zip(xs, ys)])
    return train, test
