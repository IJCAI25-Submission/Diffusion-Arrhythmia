from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../../multiBaselines')
sys.path.append('../../')
from Baselines.Models.newNets2D import *
from Baselines.EffNet.fre_load_data import gerMyDataset as MyDataset
import time
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
from copy import deepcopy


def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--formal_epochs', type=int, default=100)
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--model', type=str, default='Eff')
    parser.add_argument('--dataset', type=str, default='CPSC2018')
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


def formal_train_and_test(dataset, train_dataset, formal_epochs, device, classifier, logPath, loss_fn):
    set_rand_seed(args.random_seed)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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

    '''
    hyper parameters
    '''
    model = deepcopy(classifier)
    model.to(device)
    # loss_fn = nn.CrossEntropyLoss()
    learning_rate = args.formal_lr
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min=0, last_epoch=-1)
    train_step = 0
    test_step = 0
    epoch = formal_epochs

    '''
    train start
    '''
    for i in range(epoch):
        print('-----epoch{}-----'.format(i))
        file.write('-----epoch{}-----'.format(i) + '\n')

        model.train()
        run_loss = 0.0
        for data in train_dataloader:
            images, targets = data
            output = model(images.to(device))
            loss = loss_fn(output, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            run_loss += loss.item()
            if train_step % 100 == 0:
                print('train_step:{},loss:{}'.format(train_step, loss.item()))
                file.write('train_step:{},loss:{}'.format(train_step, loss.item()) + '\n')
        scheduler.step()
        one_epoch_loss = run_loss / len(train_dataloader)
        print('epoch{} loss:{}'.format(i, one_epoch_loss))
        file.write('epoch{} loss:{}'.format(i, one_epoch_loss) + '\n')
    model_save_path = './savedModels/pre{}_{}.pth'.format(args.model, args.dataset)
    torch.save(model, model_save_path)


def make_classifier_model(baseModel, dataset):
    if dataset == 'CPSC2018':
        classifier_model = downStreamClassifier(baseModel, num_classes=9)
    elif dataset == 'ChapmanShaoxing':
        classifier_model = downStreamClassifier(baseModel, num_classes=4)
    elif dataset == 'PTB':
        classifier_model = downStreamClassifier(baseModel, num_classes=7)

    return classifier_model


def main():
    modelName = args.model
    Name = 'pre_freBase_[model@{}]'.format(modelName)
    set_rand_seed(args.random_seed)
    dataset = args.dataset
    logPath = './pre_logs/{}/{}/{}'.format(modelName, Name, dataset)

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    classifier_model = make_classifier_model(EfficientNetB4(), dataset)

    loss_fn = nn.CrossEntropyLoss()

    train_dataset = MyDataset(dataset, '../../Dataset/data_' + dataset + '_segments/person_all_idx.txt')
    train_data_size = len(train_dataset)
    print("The length of train set: {}".format(train_data_size))

    formal_train_and_test(dataset, train_dataset, args.formal_epochs, device, classifier_model,logPath, loss_fn)

if __name__ == '__main__':
    main()
