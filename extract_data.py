def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset


class ExperimentDataset(Dataset):
    FILE_PATH = "./data/cifar10/cifar-10-batches-py/test_batch"

    def __init__(self, filepath=FILE_PATH):
        self.filepath = filepath
        d = unpickle(self.filepath)
        del d[b"batch_label"]
        del d[b"filenames"]

        k1, k2 = b"labels", b"data"

        d = sorted(tuple(zip(d[k1], d[k2])), key=lambda x: x[0])
        self.dataset = {}
        for i in range(10):
            self.dataset[i] = [
                np.transpose(
                    np.array(j[1]).reshape(3, 32, 32),
                    (1, 2, 0),
                )
                for j in d[(1000 * i) : (1000 * i + 10)]
            ]

    def __getitem__(self, cls, idx):
        img = self.dataset[cls][idx]
        return cls, img
