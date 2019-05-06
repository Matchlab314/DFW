import functools
import numpy as np
from PIL import Image
import dfw


def load_mnist():
    with open('train-images-idx3-ubyte', 'rb') as f:
        f.read(16)
        train_images = np.empty((60000, 28*28), np.uint8)
        for i, raw_image in enumerate(iter(functools.partial(f.read, 28*28), b'')):
            train_images[i] = np.frombuffer(raw_image, np.uint8)
    with open('train-labels-idx1-ubyte', 'rb') as f:
        f.read(8)
        train_labels = np.frombuffer(f.read(), np.uint8)
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        f.read(16)
        test_images = np.empty((10000, 28*28), np.uint8)
        for i, raw_image in enumerate(iter(functools.partial(f.read, 28*28), b'')):
            test_images[i] = np.frombuffer(raw_image, np.uint8)
    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        f.read(8)
        test_labels = np.frombuffer(f.read(), np.uint8)
    return train_images, train_labels, test_images, test_labels


def to_one_hot(t):
    result = np.zeros((t.shape[0], 10), np.uint8)
    for i in range(t.shape[0]):
        result[i][t[i]] = 1
    return result


def show_image(img):
    pil_img = Image.fromarray(img)
    pil_img.show()


learning_rate = 0.0001
alpha = 0.9
batch_size = 100
evaluation_size = 1000
epoch_num = 1000
print_interval = 100  # unit : iteration

train_images, train_labels, test_images, test_labels = load_mnist()

layer1 = dfw.Layer(dfw.Conv((1, 28, 28), (30, 1, 5, 5), (30, 24, 24), 0, 1, dfw.he, dfw.Momentum(learning_rate, alpha)), dfw.ReLU(), dfw.Pooling(2))
layer2 = dfw.CNN_to_FC()
layer3 = dfw.Layer(dfw.Affine(30*12*12, 100, dfw.he, dfw.Momentum(learning_rate, alpha)), dfw.ReLU())
layer4 = dfw.LastLayer(dfw.Affine(100, 10, dfw.xavier, dfw.Momentum(learning_rate, alpha)), dfw.Softmax(), dfw.SoftmaxCrossEntropy())
net = dfw.Net(layer1, layer2, layer3, layer4)

for epoch in range(epoch_num):
    for i in range(60000 // batch_size):
        index = np.random.choice(60000, batch_size)
        x = train_images[index].reshape(batch_size, 1, 28, 28)
        t = to_one_hot(train_labels[index])
        if i % print_interval == 0:
            index_test = np.random.choice(10000, evaluation_size)
            x_test = test_images[index_test].reshape(evaluation_size, 1, 28, 28)
            t_test = to_one_hot(test_labels[index_test])
            print('train loss :', net.loss(x, t))
            print('train accuracy :', net.accuracy(x, t))
            print('test loss :', net.loss(x_test, t_test))
            print('test accuracy :', net.accuracy(x_test, t_test))
        net.loss(x, t)
        net.backward()
        net.update()