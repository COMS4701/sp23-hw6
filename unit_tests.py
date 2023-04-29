from network import Network
import hydra
from omegaconf import DictConfig
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from utils import generate_test_train_val_data


def create_simple_net_fixed():
    sizes = [2, 2, 2]
    net = Network(sizes)
    net.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    net.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    return net


def test_feedforward():
    net = create_simple_net_fixed()

    gt_z = np.array([[-1.5586694], [0.80512013]])
    gt_a = np.array([[0.88594343], [0.39667158]])

    input = np.ones((2, 1))
    z, a = net.feedforward(input)

    assert np.isclose(gt_a, a[1], atol=1e-05).all()
    assert np.isclose(gt_z, z[1], atol=1e-05).all()

    print("feedforward test passed!")


def test_backprop():
    net = create_simple_net_fixed()

    input = np.ones([2, 1]).reshape(2, 1)
    output = np.array([0, 1]).reshape(2, 1)
    z, a = net.feedforward(input)
    nabla_b, nabla_w = net.backprop(input, output, z, a)

    gt_b = np.array([[0.02111149], [-0.0188499]])
    gt_w = np.array([[0.02111149, 0.02111149], [-0.0188499,  -0.0188499]])
    assert np.isclose(gt_b, nabla_b[0], atol=1e-05).all()
    assert np.isclose(gt_w, nabla_w[0], atol=1e-05).all()

    print("backprop test passed!")


def test_training(configs):
    print("testing training, this will take a minute...")

    net = Network(configs.network_size)

    # training
    train_data, val_data = generate_test_train_val_data()
    losses = net.SGD(
        train_data,
        configs.epochs,
        configs.lr,
        configs.decay,
        configs.batch_size,
        test=val_data,
    )

    # testing
    print(f"final test accuracy: {net.evaluate(val_data)}")
    test_x, test_y = np.array([d[0][0] for d in val_data]), np.array(
        [d[0][1] for d in val_data]
    )
    output = np.array(
        [np.argmax(net.feedforward(data[0])[1][-1]) for data in val_data],
        dtype=np.bool8,
    )

    part1_x = test_x[output].flatten()
    part1_y = test_y[output].flatten()
    part2_x = test_x[np.logical_not(output)]
    part2_y = test_y[np.logical_not(output)]

    fig = plt.figure(figsize=(8, 4.5))
    axes = fig.add_subplot(1, 2, 1)
    axes.plot(losses)

    axes = fig.add_subplot(1, 2, 2)
    plt.scatter(part1_x, part1_y, s=0.75)
    plt.scatter(part2_x, part2_y, s=0.75)

    plt.show()


@hydra.main(version_base="1.2", config_path="configs", config_name="unit_test.yaml")
def main(configs: DictConfig):
    test_feedforward()
    test_backprop()
    test_training(configs)


if __name__ == "__main__":
    main()
