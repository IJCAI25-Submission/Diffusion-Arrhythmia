from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../../multiBaselines')
sys.path.append('../../')
from Baselines.Models.newNets2D import *
from Baselines.EffNet.fre_load_data import gerMyDataset as MyDataset
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import argparse
import torch
import torch.nn as nn
from copy import deepcopy


def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=3, help="device")
    parser.add_argument('--formal_epochs', type=int, default=20)
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--model', type=str, default='Eff')
    parser.add_argument('--preDataset', type=str, default='CPSC2018')
    parser.add_argument('--downDataset', type=str, default='ChapmanShaoxing')
    parser.add_argument('--mode', type=str, default='train')
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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
        if i == epoch - 1:
            save_path = './savedModels/{}/{}.pth'.format(args.model, args.downDataset)
            if not os.path.exists('./savedModels/{}'.format(args.model)):
                os.makedirs('./savedModels/{}'.format(args.model))
            torch.save(model, save_path)

    file.close()

def test(dataset, test_dataset, device, classifier, logPath):
    model = deepcopy(classifier)
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    '''
    log
    '''
    if not os.path.exists(logPath + '/logs/'):
        os.makedirs(logPath + '/logs/')
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}.txt'.format(t), 'a')
    resultList = []
    labelList = []
    model.eval()
    with torch.no_grad():
        test_one_epoch_loss = 0.0
        for data in test_dataloader:
            images, targets = data
            output = model(images.to(device))
            _, predicted = torch.max(output, 1)
            resultList.extend(predicted.cpu().numpy())
            labelList.extend(targets.cpu().numpy())
        CLabelList = np.array(labelList)
        CResultList = np.array(resultList)
        C = confusion_matrix(CLabelList, CResultList)
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
            pre[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            sen[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * pre[i] * sen[i] / (pre[i] + sen[i]) if (pre[i] + sen[i]) > 0 else 0
        # print(C)
        acc = np.sum(C.diagonal()) / np.sum(C)
        macro_sen = np.mean(sen)
        macro_pre = np.mean(pre)
        macro_f1 = np.mean(f1)
        print('Accuracy:{}'.format(acc))
        print('Sensitivity:{}'.format(macro_sen))
        print('Precision:{}'.format(macro_pre))
        print('F1_score:{}'.format(macro_f1))
        file.write('Accuracy:{}\n'.format(acc))
        file.write('Sensitivity:{}\n'.format(macro_sen))
        file.write('Precision:{}\n'.format(macro_pre))
        file.write('F1_score:{}\n'.format(macro_f1))


def make_classifier_model(baseModel, dataset):
    if dataset == 'CPSC2018':
        classifier_model = downStreamClassifier(baseModel, num_classes=9)
    elif dataset == 'ChapmanShaoxing':
        classifier_model = downStreamClassifier(baseModel, num_classes=4)
    elif dataset == 'PTB':
        classifier_model = downStreamClassifier(baseModel, num_classes=2)
    elif dataset == 'Georgia':
        classifier_model = downStreamClassifier(baseModel, num_classes=4)

    return classifier_model


def main():
    modelName = args.model
    Name = '[mode@{}]_down_freBase_[model@{}]_[preDataset@{}]_[downDataset@{}]'.format(args.mode, modelName,args.preDataset,args.downDataset)
    set_rand_seed(args.random_seed)
    dataset = args.downDataset
    logPath = './ger_logs/{}/{}/{}'.format(modelName, Name, dataset)

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    baseModel = torch.load('./savedModels/pre{}_{}.pth'.format(args.model, args.preDataset), map_location=device)

    loss_fn = nn.CrossEntropyLoss()

    train_dataset = MyDataset(dataset, '../../Dataset/data_' + dataset + '_segments/person_train_idx.txt')
    test_dataset = MyDataset(dataset, '../../Dataset/data_' + dataset + '_segments/person_test_idx.txt')
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("The length of train set: {}".format(train_data_size))
    print("The length of test set: {}".format(test_data_size))

    if args.mode == 'train':
        classifier_model = make_classifier_model(baseModel.baseModel, dataset)
        formal_train_and_test(dataset, train_dataset, args.formal_epochs, device, classifier_model,logPath, loss_fn)
    elif args.mode == 'test':
        classifier_model = torch.load('./savedModels/{}/{}.pth'.format(args.model, dataset), map_location=device)
        test(dataset, test_dataset, device, classifier_model, logPath)

if __name__ == '__main__':
    main()
