import numpy as np
import matplotlib.pyplot as plt
import kagglehub

def set_data():
    path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

    return path

def ReLU(x):
    return np.maximum(0, x)

# derivative ReLU
def dReLU(x):
    return x > 1

def softmax(x):
    out = np.exp(x) / sum(np.exp(x))
    return out

def one_hot_encoding(Y):
    # (num_training, num_classes)
    # (60000, 10)
    one_hot = np.zeros((Y.size, Y.max()+1))

    # one_hot[0][Y]=1, one_hot[1][Y]=1, ..., one_hot[59999][Y]=1
    one_hot[np.arange(Y.size), Y] = 1

    # (10, 60000)
    one_hot = one_hot.transpose(-1, 0)

    return one_hot

def get_pred(P):
    return np.argmax(P, 0)

def get_acc(pred, Y):
    return (np.sum(pred == Y) / Y.size)

def visualize_parameters(W1, b1, W2, b2):
    # W2: (10, 256) -> 10개의 16x16 이미지로 시각화
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        ax = axes[i]
        img = W2[i].reshape(16, 16)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'W2-{i}')
    plt.show()
    