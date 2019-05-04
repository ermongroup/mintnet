from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from PIL import Image


class ImageNet(Dataset):
    def unpickle(self, filename):
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.data = []
            self.labels = []
            for i in range(10):
                filename = os.path.join(self.root, 'train_data_batch_{}'.format(i + 1))
                d = self.unpickle(filename)
                self.data.append(d['data'])
                self.labels.extend(d['labels'])
            self.data = np.concatenate(self.data, axis=0)
        else:
            filename = os.path.join(self.root, 'val_data')
            d = self.unpickle(filename)
            self.data = d['data']
            self.labels = d['labels']

        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.labels = np.stack(self.labels, axis=0)
        self.labels -= 1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
