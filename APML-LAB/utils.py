import numpy as np

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import torch
import torchvision
import torchvision.transforms as transforms

import math
import itertools


def MNIST_datasets(root='./data'):
    # define transformations
    flat_tensor_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Lambda(
                                                    lambda x: x.view(-1))])
    # training data set
    train_dataset = torchvision.datasets.MNIST(root=root,
                                               train=True,
                                               download=True,
                                               transform=flat_tensor_transform)

    # test data set
    test_dataset = torchvision.datasets.MNIST(root=root,
                                              train=False,
                                              download=True,
                                              transform=flat_tensor_transform)

    return train_dataset, test_dataset


def getall(dataset):
    dataset_size = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size)

    return next(iter(data_loader))


def MNIST(test=False):
    # load datasets
    train_dataset, test_dataset = MNIST_datasets()

    if test:
        return getall(test_dataset)
    else:
        return getall(train_dataset)


def plot_image(tensor, label=None):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if label:
        plt.title(f"Label: {label}", fontweight='bold')
    plt.imshow(tensor.view(28, 28).cpu().numpy(),
               vmin=0.0, vmax=1.0, cmap='gray_r')
    plt.show()


def plot_images(images_or_dataset, labels=None, shuffle=False, nrow=5,
                num_samples=25):
    # compute number of rows and columns of grid
    num_cols = min(nrow, num_samples)
    num_rows = int(math.ceil(float(num_samples) / nrow))

    # create data loader
    if isinstance(images_or_dataset, torch.utils.data.Dataset):
        dataset = images_or_dataset
    else:
        if labels is not None:
            dataset = torch.utils.data.TensorDataset(images_or_dataset, labels)
        else:
            dataset = torch.utils.data.TensorDataset(images_or_dataset)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle)

    plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    for (i, sample) in enumerate(itertools.islice(dataloader, num_samples)):
        # extract labels if provided
        if len(sample) == 1:
            image = sample[0]
            label = None
        elif len(sample) == 2:
            image, label = sample

        # configure subplot
        plt.subplot(num_rows, num_cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if label is not None:
            plt.title(f"Label: {label.item()}", fontweight='bold')

        # plot image
        plt.imshow(image.view(28, 28).cpu().numpy(),
                   vmin=0.0, vmax=1.0, cmap='gray_r')
    plt.show()


def plot_encoding(train_data, test_data, Z=None,
                  annotate=False,
                  titles=["training data", "test data"]):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    for (data, title, ax) in zip([train_data, test_data], titles, axes.flat):
        encodings, labels = data
        scatter = ax.scatter(encodings[:, 0], encodings[:, 1],
                             c=labels, cmap=plt.cm.tab10, vmin=-0.5,
                             vmax=9.5, alpha=0.7)
        if Z is not None:
            ax.scatter(Z[:, 0], Z[:, 1], c='black', marker='x')
            if annotate:
                for j in range(Z.size(0)):
                    ax.annotate(j, (Z[j, 0], Z[j, 1]))
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_title(title)

    cb = fig.colorbar(scatter, ticks=np.arange(
        0, 10), ax=axes.ravel().tolist())
    cb.ax.set_title("digit")

    plt.show()


def plot_reconstruction(images, reconstructions, labels, shuffle=False, nrow=2,
                        num_samples=8):
    # compute number of rows and columns of grid
    num_cols = min(nrow, num_samples)
    num_rows = int(math.ceil(float(num_samples) / nrow))

    # create data loader
    dataset = torch.utils.data.TensorDataset(images, reconstructions, labels)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle)

    plt.figure(figsize=(5 * num_cols, 2.5 * num_rows))
    for (i, (original, reconstruction, label)) in enumerate(
            itertools.islice(dataloader, num_samples)):
        # configure subplot
        plt.subplot(num_rows, num_cols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(f"Label: {label.item()}", fontweight='bold')

        # plot original and reconstructed image
        # design a grid, similar to torchvision.utils.make_grid
        grid = torch.ones(1, 1, 32, 62)
        grid.narrow(2, 2, 28).narrow(3, 2, 28).copy_(
            original.view(-1, 1, 28, 28))
        grid.narrow(2, 2, 28).narrow(3, 32, 28).copy_(
            reconstruction.view(-1, 1, 28, 28))
        plt.imshow(grid.squeeze().cpu().numpy(),
                   vmin=0.0, vmax=1.0, cmap='gray_r')

    plt.show()


def mean_encodings(encodings, labels):
    # compute mean encodings
    mean_encodings = []
    for i in range(10):
        mean_encoding = torch.mean(encodings[labels == i, :], dim=0)
        mean_encodings.append(mean_encoding)
    mean_encodings = torch.stack(mean_encodings, dim=0)

    return mean_encodings


def create_grid(start, stop, length=5):
    stacked = torch.stack(torch.meshgrid(
        [torch.linspace(xi, yi, length) for (xi, yi) in zip(start, stop)]),
        dim=-1).transpose(1, 0)

    return stacked.reshape(-1, stacked.size(-1))
