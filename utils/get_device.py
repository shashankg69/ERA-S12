import torch
import torchinfo
from matplotlib import pyplot as plt

SEED = 42
DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print("Device Selected:", DEVICE)
    return DEVICE


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if get_device() == 'cuda':
        torch.cuda.manual_seed(seed)

def get_incorrect_preds(prediction, labels):
        prediction = prediction.argmax(dim=1)
        indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
        return indices, prediction[indices].tolist(), labels[indices].tolist()


def plot_examples(images, labels, figsize=None):
    _ = plt.figure(figsize=figsize)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        image = images[i]
        plt.imshow(image, cmap='gray')
        label = labels[i]
        plt.title(str(label))
        plt.xticks([])
        plt.yticks([])


