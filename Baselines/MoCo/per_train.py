import torch
from torch.utils.data import DataLoader
from moco_load_data import perMyDataset as MyDataset
import os
import sys
sys.path.append('../..')
from Baselines.Models.MultiMoCoCreator import multiChannelDownStreamClassifier, MoCo1D, MoCo2D
import time
import numpy as np
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import warnings
import random
import argparse
from copy import deepcopy
import multiprocessing
from multiprocessing import Process
from torch.utils.data import Subset

warnings.filterwarnings("ignore")


def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--formal_epochs', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--split_num', type=int, default=8)
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


def listdir(path):
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)

    return list_name

def formal_train_and_test(dataset, train_dataset, formal_epochs, person, device,
                          classifier, logPath, loss_fn):
    set_rand_seed(args.random_seed)

    person = person.split('/')[-1]
    formal_train_data_size = len(train_dataset)
    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/' + person):
        os.makedirs(logPath + '/logs/' + person)
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}/{}.txt'.format(person, t), 'a')
    train_data_size = len(train_dataset)
    print('The size of train_data:{}'.format(train_data_size))
    file.write('The size of train_data:{}'.format(train_data_size) + '\n')
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
        print('-----epoch{}-----'.format(i))
        file.write('-----epoch{}-----'.format(i) + '\n')
        model.train()
        one_epoch_loss = 0.0
        train_acc = 0
        for data in train_dataloader:
            singal, lorenz, frequency, targets = data
            output = model(singal.to(device), lorenz.to(device), frequency.to(device))
            loss = loss_fn(output, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            one_epoch_loss += loss.item()
            current_acc = (output.argmax(1).to('cpu') == targets).sum()
            train_acc += current_acc
            if train_step % 2 == 0:
                print('person:{},train_step:{},loss:{}'.format(person, train_step, loss.item()))
                file.write('person:{},train_step:{},loss:{}'.format(person, train_step, loss.item()) + '\n')

        scheduler.step()
        acc = train_acc / (formal_train_data_size)
        print('train_acc:{}'.format(acc))

        if i == epoch - 1:
            save_path = 'savedModels/{}/classifier.pth'.format(person)
            if not os.path.exists('savedModels/{}'.format(person)):
                os.makedirs('savedModels/{}'.format(person))
            torch.save(model, save_path)
            print('Model saved in {}'.format(save_path))
    file.close()

def test(test_dataset, person, device, classifier, logPath):

    model = deepcopy(classifier)
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/' + person):
        os.makedirs(logPath + '/logs/' + person)
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}/{}.txt'.format(person, t), 'a')

    model.eval()
    accuracy = 0
    labelList = []
    resultList = []
    with torch.no_grad():
        for data in test_dataloader:
            singal, lorenz, frequency, targets = data
            output = model(singal.to(device), lorenz.to(device), frequency.to(device))
            predict = output.argmax(1)
            current_acc = (output.argmax(1).to('cpu') == targets).sum()
            accuracy += current_acc

            for j in range(len(predict)):
                if targets[j] == 1:
                    labelList.append('1')
                else:
                    labelList.append('0')

                if predict[j] == 1:
                    resultList.append('1')
                else:
                    resultList.append('0')

    C_labelList = np.array(labelList).astype(int)
    C_resultList = np.array(resultList).astype(int)
    C = confusion_matrix(C_labelList, C_resultList, labels=[0, 1])
    # print(C)
    sensitivity = C[1][1] / (C[1][1] + C[1][0])
    acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])

    precision = C[1][1] / (C[1][1] + C[0][1])
    specificity = C[0][0] / (C[0][0] + C[0][1])

    f1 = (2 * precision * sensitivity) / (precision + sensitivity)

    print('-------Confusion Matrix-------')
    print('TN:{}, FP:{}'.format(C[0][0], C[0][1]))
    print('FN:{}, TP:{}'.format(C[1][0], C[1][1]))
    file.write('-------Confusion Matrix-------' + '\n')
    file.write('TN:{}, FP:{}'.format(C[0][0], C[0][1]) + '\n')
    file.write('FN:{}, TP:{}'.format(C[1][0], C[1][1]) + '\n')
    print('-------eval accuracy:{}'.format(acc))
    file.write('-------eval accuracy:{}'.format(acc) + '\n')
    print('-------eval Sensitivity:{}'.format(sensitivity))
    file.write('-------eval Sensitivity:{}'.format(sensitivity) + '\n')
    print('-------eval Specificity:{}'.format(specificity))
    file.write('-------eval Specificity:{}'.format(specificity) + '\n')
    print('-------eval Precision:{}'.format(precision))
    file.write('-------eval Precision:{}'.format(precision) + '\n')
    print('-------eval F1_score:{}\n'.format(f1))
    file.write('-------eval F1_score:{}'.format(f1) + '\n')
    file.close()

def create_dataset(person, dataset):
    set_rand_seed()
    if args.mode == 'train':
        data_transform = {
            "train": transforms.Compose([transforms.Resize(48),
                                         transforms.ToTensor()]),
            "val": transforms.Compose([transforms.Resize(48),
                                       transforms.ToTensor()])}

        train_dataset = MyDataset(person, dataset,
                                  '../../Dataset/data_' + dataset + '_segments/personalIdx/' + person + '/train_idx_19.txt',
                                  mode='train', transform=data_transform['train'])
        test_dataset = MyDataset(person, dataset,
                                 '../../Dataset/data_' + dataset + '_segments/personalIdx/' + person + '/test_idx_19.txt',
                                 mode='test', transform=data_transform['val'])

        class_samples = {0: 0, 1: 0}
        for _, _, _, label in train_dataset:
            class_samples[label] += 1

        sample_counts = {label: int(count / 2) for label, count in class_samples.items()}
        print(sample_counts)

        weights = [1, 1]

        train_indices_to_sample = []
        val_indices_to_sample = []
        all_indices_collected = {0: [], 1: []}

        for i, (_, _, _, label) in enumerate(train_dataset):
            all_indices_collected[label].append(i)

        for label, indices in all_indices_collected.items():
            random.shuffle(indices)
            train_sampled_indices = indices[:sample_counts[label]]
            val_sampled_indices = indices[sample_counts[label]:2 * sample_counts[label]]
            train_indices_to_sample.extend(train_sampled_indices)
            val_indices_to_sample.extend(val_sampled_indices)

        val_dataset = Subset(train_dataset, val_indices_to_sample)
        new_train_dataset = Subset(train_dataset, train_indices_to_sample)

        return new_train_dataset, val_dataset, test_dataset
    elif args.mode == 'test':
        data_transform = {
            "test": transforms.Compose([transforms.Resize(48),
                                        transforms.ToTensor()])}

        test_dataset = MyDataset(person, dataset,
                                 '../../Dataset/data_' + dataset + '_segments/personalIdx/' + person + '/test_idx_19.txt',
                                 mode='test', transform=data_transform['test'])

        return test_dataset


def process_individual(person, args, logPath, device, classifer_model, dataset, loss_fn):
    set_rand_seed(args.random_seed)
    print("person:{}".format(person))

    if args.mode == 'train':
        train_dataset, val_dataset, test_dataset= create_dataset(person, dataset)
        train_data_size = len(train_dataset)
        val_dataset_size = len(val_dataset)
        test_data_size = len(test_dataset)
        print("The length of train set: {}".format(train_data_size))
        print("The length of val set: {}".format(val_dataset_size))
        print("The length of test set: {}".format(test_data_size))
        formal_train_and_test(dataset, train_dataset, args.formal_epochs,person, device, classifer_model, logPath, loss_fn)
    elif args.mode == 'test':
        test_dataset = create_dataset(person, dataset)
        print("The length of test set: {}".format(len(test_dataset)))
        classifer_model = torch.load('./savedModels/{}/classifier.pth'.format(person), map_location=device)
        test(test_dataset, person, device, classifer_model, logPath)


def main():
    Name = 'MoCo_[mode@{}]_per'.format(args.mode)
    set_rand_seed(args.random_seed)
    dataset = "LTAF"
    logPath = './per_logs/{}/{}'.format(Name, dataset)

    list = listdir('../../Dataset/data_' + dataset + '_segments/personalIdx/')
    list.sort()
    person_list = list

    split_num = args.split_num
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    person_groups = [person_list[i::split_num] for i in range(split_num)]
    signal_encoder = MoCo1D(device=device, base_model='unet', view='signal')
    frequency_encoder = MoCo2D(device=device, base_model='unet', view='frequency')
    lorentz_encoder = MoCo2D(device=device, base_model='unet', view='lorenz')
    signal_encoder.load_state_dict(torch.load('./savedModels/MoCoModel_CPSC2018_unet_signal.pth', map_location=device))
    frequency_encoder.load_state_dict(torch.load('./savedModels/MoCoModel_CPSC2018_unet_frequency.pth', map_location=device))
    lorentz_encoder.load_state_dict(torch.load('./savedModels/MoCoModel_CPSC2018_unet_lorenz.pth', map_location=device))
    classifer_model = multiChannelDownStreamClassifier(
        signal_encoder.encoder_q,
        frequency_encoder.encoder_q,
        lorentz_encoder.encoder_q,
        num_classes=2
    )
    loss_fn = nn.CrossEntropyLoss()

    processes = []
    for group in person_groups:
        for person in group:
            person = person.split('/')[-1]
            p = Process(target=process_individual, args=(
            person, args, logPath, device, classifer_model,
            dataset, loss_fn))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()