import torch
import sys
sys.path.append('../../multiBaselines')
sys.path.append('../../')
from torch.utils.data import Dataset
import numpy as np

class gerMyDataset(Dataset):
    def __init__(self, dataset, txt_path):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line
            line = line.rstrip()
            words = line.split()
            imgs.append(('../../Frequency/STFT/' + dataset + '/segments/' + words[0].split('/')[-1].strip(
                'txt') + 'npy', int(words[1])))

        self.imgs = imgs

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = torch.from_numpy(np.float32(np.load(fn)))
        img = img.unsqueeze(0)

        return img, label

    def __len__(self):
        return len(self.imgs)


class perMyDataset(Dataset):
    def __init__(self, person, dataset, txt_path):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line
            line = line.rstrip()
            words = line.split()
            imgs.append(('../../Frequency/STFT/' + dataset + '/' + person + '/' + words[0].split('/')[-1].strip(
                'txt') + 'npy', int(words[1])))

        self.imgs = imgs

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = torch.from_numpy(np.float32(np.load(fn)))
        img = img.unsqueeze(0)

        return img, label

    def __len__(self):
        return len(self.imgs)

