import sys
sys.path.append('../..')
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import os
import time
from copy import deepcopy
import argparse
import random
from Diffusion_based.DiffusionModels.noisePredictModels.Unet._1DUNet import Unet as _1DUnet
from Diffusion_based.DiffusionModels.noisePredictModels.Unet._2DUNet import Unet as _2DUnet
from Diffusion_based.DiffusionModels.Diffusion._1DDiffusion import DiffusionModel as _1DDM
from Diffusion_based.DiffusionModels.Diffusion._2DDiffusion import DiffusionModel as _2DDM
from classifier import CrossAttnClassifier
from load_data import gerMyDataset as MyDataset


def _lr_fn(epoch):
    if epoch < 4:
        lr = 0.0001 + (1 - 0.00001) / 4 * epoch
    else:
        e = epoch - 4
        es = 16
        lr = 0.5 * (1 + np.cos(np.pi * e / es))
    return lr

def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--formal_epochs', type=int, default=20)
    parser.add_argument('--preDataset', type=str, default='CPSC2018')
    parser.add_argument('--downDataset', type=str, default='ChapmanShaoxing')
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--mode', type=str, default='test')

    return parser.parse_args()


args = args_parse()

def set_rand_seed(seed=args.random_seed):
    # print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def formal_train_and_test(t1v, t2v, t3v, dataset, train_dataset, test_dataset, formal_epochs, device,
                          classifier, signalExtractor, lorenzExtractor, frequencyExtractor, logPath, loss_fn):
    set_rand_seed(args.random_seed)
    for param in signalExtractor.parameters():
        param.requires_grad = False
    for param in lorenzExtractor.parameters():
        param.requires_grad = False
    for param in frequencyExtractor.parameters():
        param.requires_grad = False

    mode = 'feature_extractor'
    formal_train_data_size = len(train_dataset)
    batch_size = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/'):
        os.makedirs(logPath + '/logs/')
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}.txt'.format(t), 'a')
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print('The size of train_data:{}'.format(train_data_size))
    file.write('The size of train_data:{}'.format(train_data_size) + '\n')
    print('The size of train_data:{}'.format(test_data_size))
    file.write('The size of train_data:{}'.format(test_data_size) + '\n')
    file.write('batch_size:{}'.format(batch_size) + '\n')

    model = deepcopy(classifier)
    model.to(device)
    learning_rate = args.formal_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_fn)
    train_step = 0
    epoch = formal_epochs
    '''
    train start
    '''
    for i in range(epoch):
        resultList = []
        labelList = []
        print('-----epoch{}-----'.format(i))
        file.write('-----epoch{}-----'.format(i) + '\n')

        model.train()
        one_epoch_loss = 0.0
        train_acc = 0
        print(scheduler.get_last_lr())
        for data in train_dataloader:
            trainIter_batch_size = data[0].shape[0]
            t1 = torch.Tensor([t1v] * trainIter_batch_size).long().to(device)
            t2 = torch.Tensor([t2v] * trainIter_batch_size).long().to(device)
            t3 = torch.Tensor([t3v] * trainIter_batch_size).long().to(device)
            singal, lorenz, frequency, targets = data
            singal_feature = signalExtractor(mode=mode, x_start=singal.to(device), t=t1)
            lorenz_feature = lorenzExtractor(mode=mode, x_start=lorenz.to(device), t=t2)
            frequency_feature = frequencyExtractor(mode=mode, x_start=frequency.to(device), t=t3)

            output = model(singal_feature.to(device), lorenz_feature.to(device), frequency_feature.to(device))
            loss = loss_fn(output, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            one_epoch_loss += loss.item()
            predict = output.argmax(1)
            current_acc = (output.argmax(1).to('cpu') == targets).sum()
            train_acc += current_acc
            if train_step % 50 == 0:
                print('train_step:{},loss:{}'.format(train_step, loss.item()))
                file.write('train_step:{},loss:{}'.format(train_step, loss.item()) + '\n')
        scheduler.step()
        acc = train_acc / (formal_train_data_size)
        print('train_acc:{}'.format(acc))
        if i == epoch - 1:
            save_path = 'savedModels/{}/classifier.pth'.format(dataset)
            if not os.path.exists('savedModels/{}'.format(dataset)):
                os.makedirs('savedModels/{}'.format(dataset))
            torch.save(model, save_path)

    file.close()

def test(t1v, t2v, t3v, dataset, test_dataset, device, classifier, signalExtractor, lorenzExtractor, frequencyExtractor, logPath):
    set_rand_seed(args.random_seed)
    for param in signalExtractor.parameters():
        param.requires_grad = False
    for param in lorenzExtractor.parameters():
        param.requires_grad = False
    for param in frequencyExtractor.parameters():
        param.requires_grad = False

    mode = 'feature_extractor'
    batch_size = 64

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/'):
        os.makedirs(logPath + '/logs/')
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}.txt'.format(t), 'a')

    model = deepcopy(classifier)
    model.to(device)
    accuracy = 0
    resultList = []
    labelList = []

    with torch.no_grad():
        for data in test_dataloader:
            valIter_batch_size = data[0].shape[0]
            val_t1 = torch.Tensor([t1v] * valIter_batch_size).long().to(device)
            val_t2 = torch.Tensor([t2v] * valIter_batch_size).long().to(device)
            val_t3 = torch.Tensor([t3v] * valIter_batch_size).long().to(device)
            singal, lorenz, frequency, targets = data
            singal_feature = signalExtractor(mode=mode, x_start=singal.to(device), t=val_t1)
            lorenz_feature = lorenzExtractor(mode=mode, x_start=lorenz.to(device), t=val_t2)
            frequency_feature = frequencyExtractor(mode=mode, x_start=frequency.to(device), t=val_t3)

            output = model(singal_feature.to(device), lorenz_feature.to(device), frequency_feature.to(device))
            predict = output.argmax(1)
            current_acc = (output.argmax(1).to('cpu') == targets).sum()
            accuracy += current_acc
            resultList.extend(predict.to('cpu').numpy())
            labelList.extend(targets.to('cpu').numpy())

    C_labelList = np.array(labelList).astype(int)
    C_resultList = np.array(resultList).astype(int)
    C = confusion_matrix(C_labelList, C_resultList)
    print('Confusion Matrix:')
    print(C)
    file.write('Confusion Matrix:' + '\n')
    file.write(str(C) + '\n')
    num_classes = C.shape[0]
    pre = np.zeros(num_classes)
    sen = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for i in range(num_classes):
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp
        precison = tp / (tp + fp) if (tp + fp) > 0 else 0
        sen[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precison * sen[i] / (precison + sen[i]) if (precison + sen[i]) > 0 else 0
    # print(C)
    acc = np.sum(C.diagonal()) / np.sum(C)
    macro_sen = np.mean(sen)
    macro_pre = np.mean(pre)
    macro_f1 = np.mean(f1)

    print('-------eval accuracy:{}'.format(acc))
    file.write('-------eval accuracy:{}'.format(acc) + '\n')
    print('-------eval Sensitivity:{}'.format(macro_sen))
    file.write('-------eval Sensitivity:{}'.format(macro_sen) + '\n')
    print('-------eval Precision:{}'.format(macro_pre))
    file.write('-------eval Precision:{}'.format(macro_pre) + '\n')
    print('-------eval F1_score:{}'.format(macro_f1))
    file.write('-------eval F1_score:{}'.format(macro_f1) + '\n')

    file.close()
def init_feature_extractors(device):
    '''
    hyper parameters
    '''
    signal_channels = 1
    signal_dim_mults = (1, 2, 2,)

    lorenz_channels = frequency_channels = 3
    lorenz_dim_mults = (1, 2, 4)
    frequency_dim_mults = (1, 2, 2, 2,)

    mode = 'encoder_only'
    timesteps = 1000
    schedule_name = "linear_beta_schedule"

    '''
    init denoise models
    '''
    signalDenoise_model = _1DUnet(
        dim=48,
        channels=signal_channels,
        dim_mults=signal_dim_mults,
        mode=mode
    )
    lorenzDenoise_model = _2DUnet(
        dim=48,
        channels=lorenz_channels,
        dim_mults=lorenz_dim_mults,
        mode=mode
    )
    frequencyDenoise_model = _2DUnet(
        dim=64,
        channels=frequency_channels,
        dim_mults=frequency_dim_mults,
        mode=mode
    )

    '''
    init feature extractors
    '''
    signalExtractor = _1DDM(schedule_name=schedule_name,
                               timesteps=timesteps,
                               beta_start=0.0001,
                               beta_end=0.02,
                               denoise_model=signalDenoise_model).to(device)
    lorenzExtractor = _2DDM(schedule_name=schedule_name,
                               timesteps=timesteps,
                               beta_start=0.0001,
                               beta_end=0.02,
                               denoise_model=lorenzDenoise_model).to(device)
    frequencyExtractor = _2DDM(schedule_name=schedule_name,
                                     timesteps=timesteps,
                                     beta_start=0.0001,
                                     beta_end=0.02,
                                     denoise_model=frequencyDenoise_model).to(device)

    lorenzExtractor.load_state_dict(torch.load('./savedModels/extractor_lorenz_{}.pth'.format(args.preDataset), map_location=device))
    frequencyExtractor.load_state_dict(torch.load('./savedModels/extractor_frequency_{}.pth'.format(args.preDataset), map_location=device))
    signalExtractor.load_state_dict(torch.load('./savedModels/extractor_signal_{}.pth'.format(args.preDataset), map_location=device))

    return signalExtractor, lorenzExtractor, frequencyExtractor


def create_dataset(dataset):
    set_rand_seed()
    data_transform = {
        "train": transforms.Compose([transforms.Resize(48),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize(48),
                                   transforms.ToTensor()])}

    train_dataset = MyDataset(dataset,
                              '../../Dataset/data_' + dataset + '_segments/person_train_idx.txt', transform=data_transform['train'])
    test_dataset = MyDataset(dataset,
                             '../../Dataset/data_' + dataset + '_segments/person_test_idx.txt', transform=data_transform['val'])


    return train_dataset, test_dataset

def main():
    Name = '[mode@{}]_ger_[predataset@{}]_[downdataset@{}]'.format(
        args.mode,
        args.preDataset,
        args.downDataset
    )
    set_rand_seed(args.random_seed)
    dataset = args.downDataset
    logPath = './ger_logs/{}/{}'.format(Name, dataset)

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    signalExtractor, lorenzExtractor, frequencyExtractor = init_feature_extractors(device)
    signalExtractor.to(device)
    lorenzExtractor.to(device)
    frequencyExtractor.to(device)
    if dataset == 'ChapmanShaoxing':
        classifier_model = CrossAttnClassifier(num_classes=4).to(device)
    elif dataset == 'CPSC2018':
        classifier_model = CrossAttnClassifier(num_classes=9).to(device)
    elif dataset == 'PTB':
        classifier_model = CrossAttnClassifier(num_classes=2).to(device)
    elif dataset == 'Georgia':
        classifier_model = CrossAttnClassifier(num_classes=4).to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset, test_dataset = create_dataset(dataset)
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("The length of train set: {}".format(train_data_size))
    print("The length of test set: {}".format(test_data_size))

    best_t1v = best_t2v = best_t3v = 20

    if args.mode == 'train':
        formal_train_and_test(best_t1v, best_t2v, best_t3v, dataset, train_dataset, test_dataset, args.formal_epochs,
                          device, classifier_model, signalExtractor, lorenzExtractor, frequencyExtractor,
                          logPath, loss_fn)
    elif args.mode == 'test':
        classifier_model = torch.load('savedModels/{}/classifier.pth'.format(dataset), map_location=device)
        test(best_t1v, best_t2v, best_t3v, dataset, test_dataset, device, classifier_model, signalExtractor, lorenzExtractor, frequencyExtractor, logPath)

if __name__ == '__main__':
    main()