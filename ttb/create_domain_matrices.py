from json import load
from this import d
from dotenv import load_dotenv
from pathlib import Path
from torchvision import datasets

import numpy as np
import os
import typing

load_dotenv()
DOWNLOAD_PREFIX = (
    os.getenv("DOWNLOAD_PREFIX")
    if os.getenv("DOWNLOAD_PREFIX")
    else "streams_data"
)
HOME = str(Path.home())


def get_mnist():
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "mnist")
    mnist_train = datasets.MNIST(download_path, train=True, download=True)
    mnist_test = datasets.MNIST(download_path, train=False, download=True)

    # Concate train and test
    dataset = mnist_train + mnist_test

    # There will be 1 domain and 10 values (labels)
    domain_matrix = np.zeros((len(dataset), 10))
    for idx, elem in enumerate(dataset):
        domain_matrix[idx][elem[1]] = 1

    return dataset, [domain_matrix]


name_to_func = {"mnist": get_mnist}
