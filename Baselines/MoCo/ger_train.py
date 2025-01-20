import torch
from torch.utils.data import DataLoader
from moco_load_data import gerMyDataset as MyDataset
import os
import sys
sys.path.append('../..')
from Baselines.Models.MultiMoCoCreator import multiChannelDownStreamClassifier, MoCo1D, MoCo2D
import time
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import warnings
import random
import argparse
from copy import deepcopy

def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--formal_epochs', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--preDataset', type=str, default='CPSC2018')
    parser.add_argument('--downDataset', type=str, default='ChapmanShaoxing')
    parser.add_argument('--mode', type=str, default='train')

    return parser.parse_args()

args = args_parse()

def _lr_fn(epoch):
    if epoch < 4:
        lr = 0.0001 + (1 - 0.00001) / 4 * epoch
    else:
        e = epoch - 4
        es = 16
        lr = 0.5 * (1 + np.cos(np.pi * e / es))
    return lr

def set_rand_seed(seed=args.random_seed):
    # print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def formal_train_and_test(classifier, train_dataset, formal_epochs, device, logPath, loss_fn):
    set_rand_seed(args.random_seed)
    '''
    load data
    '''
    formal_train_data_size = len(train_dataset)
    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/'):
        os.makedirs(logPath + '/logs/')
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}.txt'.format(t), 'a')
    train_data_size = len(train_dataset)
    print('The size of train_data:{}'.format(train_data_size))
    file.write('The size of train_data:{}'.format(train_data_size) + '\n')
    file.write('batch_size:{}'.format(batch_size) + '\n')

    classifier_model = deepcopy(classifier)
    classifier_model.to(device)
    learning_rate = args.formal_lr
    optimizer = torch.optim.SGD(classifier_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_fn)
    train_step = 0
    epoch = formal_epochs
    '''
    train start
    '''
    for i in range(epoch):
        print('-----epoch{}-----'.format(i))
        file.write('-----epoch{}-----'.format(i) + '\n')
        classifier_model.train()
        one_epoch_loss = 0.0
        train_acc = 0
        for data in train_dataloader:
            singals, lorenz, frequency, targets = data
            output = classifier_model(singals.to(device), lorenz.to(device), frequency.to(device))
            loss = loss_fn(output, targets.to(device))
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            one_epoch_loss += loss.item()
            current_acc = (output.argmax(1).to('cpu') == targets).sum()
            train_acc += current_acc
            if train_step % 50 == 0:
                print('train_step:{},loss:{}'.format(train_step, loss.item()))
                file.write('train_step:{},loss:{}'.format(train_step, loss.item()) + '\n')
        scheduler.step()

        acc = train_acc / (formal_train_data_size)
        print('train_acc:{}'.format(acc))

        if i == epoch - 1:
            save_path = './savedModels/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(classifier_model, save_path + '{}.pth'.format(args.downDataset))
            print('Model saved in {}'.format(save_path))
            file.write('Model saved in {}'.format(save_path) + '\n')

    file.close()


def test(classifier, test_dataset, device, logPath):
    classifier_model = deepcopy(classifier)
    classifier_model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/'):
        os.makedirs(logPath + '/logs/')
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}.txt'.format(t), 'a')
    resultList = []
    labelList = []
    classifier_model.eval()
    accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            singals, lorenz, frequency, targets = data
            output = classifier_model(singals.to(device), lorenz.to(device), frequency.to(device))
            _, predicted = output.max(1)
            current_acc = (output.argmax(1).to('cpu') == targets).sum()
            accuracy += current_acc
            resultList.extend(predicted.to('cpu').numpy())
            labelList.extend(targets.to('cpu').numpy())

    C_labelList = np.array(labelList).astype(int)
    C_resultList = np.array(resultList).astype(int)
    C = confusion_matrix(C_labelList, C_resultList)
    num_classes = C.shape[0]
    pre = np.zeros(num_classes)
    sen = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for i in range(num_classes):
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp
        pre[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        sen[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * pre[i] * sen[i] / (pre[i] + sen[i]) if (pre[i] + sen[i]) > 0 else 0
    # print(C)
    acc = np.sum(C.diagonal()) / np.sum(C)
    macro_sen = np.mean(sen)
    macro_pre = np.mean(pre)
    macro_f1 = np.mean(f1)

    print('-------Confusion Matrix-------')
    print(C)
    file.write('-------Confusion Matrix-------' + '\n')
    file.write(str(C) + '\n')
    print('-------eval accuracy:{}'.format(acc))
    file.write('-------eval accuracy:{}'.format(acc) + '\n')
    print('-------eval Sensitivity:{}'.format(macro_sen))
    file.write('-------eval Sensitivity:{}'.format(macro_sen) + '\n')
    print('-------eval Precision:{}'.format(macro_pre))
    file.write('-------eval Precision:{}'.format(macro_pre) + '\n')
    print('-------eval F1_score:{}'.format(macro_f1))
    file.write('-------eval F1_score:{}'.format(macro_f1) + '\n')

    file.close()


def create_dataset(dataset):

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    train_dataset = MyDataset(dataset, '../../Dataset/data_' + dataset + '_segments/person_train_idx.txt', transform=transform)
    test_dataset = MyDataset(dataset, '../../Dataset/data_' + dataset + '_segments/person_test_idx.txt', transform=transform)

    return train_dataset, test_dataset

def make_classifier_model(signal_encoder, frequency_encoder, lorentz_encoder, dataset):
    if dataset == 'CPSC2018':
        classifier_model = multiChannelDownStreamClassifier(signal_encoder.encoder_q, frequency_encoder.encoder_q, lorentz_encoder.encoder_q, num_classes=9)
    elif dataset == 'ChapmanShaoxing':
        classifier_model = multiChannelDownStreamClassifier(signal_encoder.encoder_q, frequency_encoder.encoder_q, lorentz_encoder.encoder_q, num_classes=4)
    elif dataset == 'PTB':
        classifier_model = multiChannelDownStreamClassifier(signal_encoder.encoder_q, frequency_encoder.encoder_q, lorentz_encoder.encoder_q, num_classes=2)
    elif dataset == 'Georgia':
        classifier_model = multiChannelDownStreamClassifier(signal_encoder.encoder_q, frequency_encoder.encoder_q, lorentz_encoder.encoder_q, num_classes=4)

    return classifier_model

def main():
    Name = 'MoCo_[mode@{}]_[preDataset@{}]_[downDataset@{}]_[formal_epochs@{}]_[formal_lr@{}]'.format(args.mode, args.preDataset, args.downDataset, args.formal_epochs, args.formal_lr)
    set_rand_seed(args.random_seed)
    dataset = args.downDataset
    logPath = './ger_logs/{}'.format(Name)

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    signal_encoder = MoCo1D(device=device, base_model='unet', view='signal')
    frequency_encoder = MoCo2D(device=device, base_model='unet', view='frequency')
    lorentz_encoder = MoCo2D(device=device, base_model='unet', view='lorenz')
    signal_encoder.load_state_dict(torch.load('./savedModels/MoCoModel_{}_unet_signal.pth'.format(args.preDataset), map_location=device))
    frequency_encoder.load_state_dict(torch.load('./savedModels/MoCoModel_{}_unet_frequency.pth'.format(args.preDataset), map_location=device))
    lorentz_encoder.load_state_dict(torch.load('./savedModels/MoCoModel_{}_unet_lorenz.pth'.format(args.preDataset), map_location=device))
    classifer_model = make_classifier_model(signal_encoder, frequency_encoder, lorentz_encoder, dataset)
    loss_fn = nn.CrossEntropyLoss()

    set_rand_seed(args.random_seed)
    train_dataset, test_dataset= create_dataset(dataset)
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("The length of train set: {}".format(train_data_size))
    print("The length of test set: {}".format(test_data_size))

    if args.mode == 'train':
        formal_train_and_test(classifer_model, train_dataset, args.formal_epochs, device, logPath, loss_fn)
    elif args.mode == 'test':
        classifier_model = torch.load('./savedModels/{}.pth'.format(args.downDataset), map_location=device)
        test(classifier_model, test_dataset, device, logPath)

if __name__ == '__main__':
    main()