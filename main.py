import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from network import Network
from utils import load_data_helper


@hydra.main(version_base="1.2", config_path="configs", config_name="mnist.yaml")
def main(configs: DictConfig):
    split = configs.train_split

    data = np.load(configs.data_path)
    x_train, x_test, y_train, y_test = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
    )
    length = len(x_train)

    x_val = x_train[int(split * length) :]
    y_val = y_train[int(split * length) :]
    x_train = x_train[: int(split * length)]
    y_train = y_train[: int(split * length)]

    training_data, validation_data, test_data = load_data_helper(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    net = Network(configs.network_size)
    losses = net.SGD(
        training_data,
        configs.epochs,
        configs.lr,
        configs.decay,
        configs.batch_size,
        test=validation_data,
    )

    print(f"final test accuracy: {net.evaluate(test_data)}")
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
