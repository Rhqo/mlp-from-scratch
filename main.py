import numpy as np
import pandas as pd
from tools import set_data, ReLU, dReLU, softmax, one_hot_encoding, get_pred, get_acc, visualize_parameters

def init_params():
    # distribution: [0, 1) -> [-0.5, 0.5)
    W1 = np.random.rand(256, 784) - 0.5
    b1 = np.random.rand(256, 1) - 0.5
    W2 = np.random.rand(10, 256) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def forward(x, W1, b1, W2, b2):
    Z1 = W1 @ x + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    P = softmax(Z2)

    return Z1, A1, Z2, P

def back_prop(Z1, A1, Z2, P, W1, W2, X, Y):
    one_hot_Y = one_hot_encoding(Y)

    dZ2 = P - one_hot_Y
    dW2 = 1 / Y.size * (dZ2 @ A1.transpose(-1, 0))
    db2 = 1 / Y.size * np.sum(dZ2)

    dZ1 = (W2.transpose(-1, 0) @ dZ2) * dReLU(Z1)
    dW1 = 1 / Y.size * (dZ1 @ X.transpose(-1, 0))
    db1 = 1 / Y.size * np.sum(dZ1)

    return dW2, db2, dW1, db1

def update_parameters(W1, b1, W2, b2, dW2, db2, dW1, db1, lr):
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    return W1, b1, W2, b2

def gradient_descent(X, Y, iterations, lr):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, P = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, P, W1, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if i % 50 == 0:
            print("Iter: ", i)
            pred = get_pred(P)
            print("Acc : ", get_acc(pred, Y))

    visualize_parameters(W1, b1, W2, b2)
    return W1, b1, W2, b2


if __name__ == '__main__':
    path = set_data()
    print("data_path: ", path)

    data = pd.read_csv(path+"/mnist_train.csv")
    data.info()

    data = np.array(data)
    data = data.transpose(-1, 0)

    # (60000, 0)
    Y_train = data[0]
    # (784, 60000)
    X_train = data[1:]
    X_train = X_train / 255

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)