# coding: utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.signal as signal
from PIL import Image

class gerMyDataset(Dataset):
    def __init__(self, dataset, txt_path, transform=None):
        fh = open(txt_path, 'r')
        inputs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            signalPath = '../../Dataset/' + words[0].split('./')[-1]
            lorenzPath = '../../Lorenz/PlotsFigure/' + dataset + '/segments/' + words[0].split('/')[-1].strip('txt') + 'jpg'
            frequencyPath = '../../Frequency/STFT/' + dataset + '/segments/' + words[0].split('/')[-1].strip('txt') + 'png'
            inputs.append((signalPath, lorenzPath, frequencyPath, int(words[1])))

        self.inputs = inputs
        self.transform = transform

    def __getitem__(self, index):
        signalPath, lorenzPath, frequencyPath, label = self.inputs[index]

        raw_signal = np.float32(np.loadtxt(signalPath))
        new_signal = torch.from_numpy(raw_signal)
        new_signal = new_signal.unsqueeze(0)

        lorenz = Image.open(lorenzPath).convert('RGB')
        if self.transform is not None:
            lorenz = self.transform(lorenz)

        frequency = Image.open(frequencyPath).convert('RGB')
        if self.transform is not None:
            frequency = self.transform(frequency)

        return new_signal, lorenz, frequency, label

    def __len__(self):
        return len(self.inputs)

class perMyDataset(Dataset):
    def __init__(self, person, dataset, txt_path, mode, transform=None):
        fh = open(txt_path, 'r')
        inputs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            signalPath = '../../Dataset/' + words[0].split('./')[-1]
            lorenzPath = '../../Lorenz/PlotsFigure/' + dataset + '/'+ person + '/' + words[0].split('/')[-1].strip('txt') + 'jpg'
            frequencyPath = '../../Frequency/STFT/' + dataset + '/'+ person + '/' + words[0].split('/')[-1].strip('txt') + 'png'
            inputs.append((signalPath, lorenzPath, frequencyPath, int(words[1])))

        self.inputs = inputs
        self.transform = transform

    def __getitem__(self, index):
        signalPath, lorenzPath, frequencyPath, label = self.inputs[index]

        raw_signal = np.float32(np.loadtxt(signalPath))
        new_signal = torch.from_numpy(raw_signal)
        new_signal = new_signal.unsqueeze(0)

        lorenz = Image.open(lorenzPath).convert('RGB')
        if self.transform is not None:
            lorenz = self.transform(lorenz)

        frequency = Image.open(frequencyPath).convert('RGB')
        if self.transform is not None:
            frequency = self.transform(frequency)

        return new_signal, lorenz, frequency, label

    def __len__(self):
        return len(self.inputs)
