# coding: utf-8
from torch.utils.data import Dataset
import numpy as np
import scipy.signal as signal
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

'''for pre-training'''
class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, signal):
        noise = torch.randn_like(signal) * self.std + self.mean
        noisy_signal = signal + noise
        return noisy_signal


# 振幅缩放
class ScaleAmplitude(object):
    def __init__(self, scale_factor=1.5):
        self.scale_factor = scale_factor

    def __call__(self, signal):
        scaled_signal = signal * self.scale_factor
        return scaled_signal


# Y轴翻转
class FlipYAxis(object):
    def __call__(self, signal):
        flipped_signal = -signal
        return flipped_signal


# X轴翻转
class FlipXAxis(object):
    def __call__(self, signal):
        flipped_signal = torch.flip(signal, [0])
        return flipped_signal

class MoCoDataset2D(Dataset):
    def __init__(self, dataset, txt_path, modality, transform=None):
        fh = open(txt_path, 'r')
        inputs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if modality == 'lorenz':
                input  = '../../Lorenz/PlotsFigure/' + dataset + '/segments/' + words[0].split('/')[-1].strip('txt') + 'jpg'
                if os.path.exists(input):
                    inputs.append((input, int(words[1])))
                else:
                    with open('lore_error.txt', 'a') as efile:
                        efile.write(words[0].split('/')[-1].strip('.txt') + '-' + words[1] + '\n')
            elif modality == 'frequency':
                input = '../../Frequency/STFT/' + dataset + '/segments/' + words[0].split('/')[-1].strip('txt') + 'png'
                if os.path.exists(input):
                    inputs.append((input, int(words[1])))
                else:
                    with open('fre_error.txt', 'a') as efile:
                        efile.write(words[0].split('/')[-1].strip('.txt') + '-' + words[1] + '\n')

        self.inputs = inputs
        self.transform = transform

    def __getitem__(self, index):
        inputPath, label = self.inputs[index]
        input = Image.open(inputPath).convert('RGB')
        if self.transform is not None:
            input1 = self.transform(input)
        input2 = self.random_augmentation(input1)

        return input1, input2

    def __len__(self):
        return len(self.inputs)

    def random_augmentation(self, input):
        img_transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=10)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(hue=0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomErasing()], p=0.5),
            transforms.RandomApply([transforms.RandomGrayscale()], p=0.5)
        ])
        augmented_input = img_transform(input)
        return augmented_input


class MoCoDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        signals = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if words[0].split('/')[2] == 'Dataset':
                signals.append((words[0], int(words[1])))
            else:
                signals.append(('../../Dataset/' + words[0], int(words[1])))

        self.signals = signals
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.signals[index]
        raw_signal = np.float32(np.loadtxt(fn))
        # original_freq = 250
        # target_freq = 48
        # T = 1 / original_freq
        # new_T = 1 / target_freq
        # duration = T * len(raw_signal)
        # sig_1 = signal.resample(raw_signal, int(duration / new_T))
        sig_1 = torch.Tensor(raw_signal)
        sig_1 = sig_1.unsqueeze(0)
        sig_2 = self.random_augmentation(sig_1)

        return sig_1, sig_2

    def __len__(self):
        return len(self.signals)

    def random_augmentation(self, signal):
        transform = transforms.Compose([
            transforms.RandomApply([AddGaussianNoise(mean=0, std=0.1)], p=0.5),
            transforms.RandomApply([ScaleAmplitude(scale_factor=1.5)], p=0.5),
            transforms.RandomApply([FlipYAxis()], p=0.5),
            FlipXAxis()
        ])
        augmented_signal = transform(signal)
        return augmented_signal

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

'''for personalization experiment'''
class perMyDataset(Dataset):
    def __init__(self, person, dataset, txt_path, mode, transform=None):
        fh = open(txt_path, 'r')
        inputs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            signalPath = '../../Dataset/' + words[0].split('./')[-1]
            lorenzPath = '../../Lorenz/PlotsFigure/' + dataset + '/' + person + '/' + words[0].split('/')[-1].strip(
                'txt') + 'jpg'
            frequencyPath = '../../Frequency/STFT/' + dataset + '/' + person + '/' + words[0].split('/')[-1].strip(
                'txt') + 'png'
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