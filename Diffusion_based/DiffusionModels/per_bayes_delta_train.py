import sys
sys.path.append('../..')
from skopt.utils import use_named_args
import numpy as np
import torch
from skopt import Optimizer
from skopt.space import Integer
from skopt.sampler import Lhs
from skopt.learning import RandomForestRegressor as SkRandomForestRegressor
from sklearn.metrics import confusion_matrix
import os
import time
from copy import deepcopy
import argparse
import random
import math
import multiprocessing
from multiprocessing import Process, Pool
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
from torchvision import transforms
from Diffusion_based.DiffusionModels.noisePredictModels.Unet._1DUNet import Unet as _1DUnet
from Diffusion_based.DiffusionModels.noisePredictModels.Unet._2DUNet import Unet as _2DUnet
from Diffusion_based.DiffusionModels.Diffusion._1DDiffusion import DiffusionModel as _1DDM
from Diffusion_based.DiffusionModels.Diffusion._2DDiffusion import DiffusionModel as _2DDM
from classifier import CrossAttnClassifier
from load_data import perMyDataset as MyDataset
import warnings
warnings.filterwarnings("ignore")


def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--bys_epochs', type=int, default=2)
    parser.add_argument('--bys_lr', type=float, default=1e-2)
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--bys_calls', type=int, default=300)
    parser.add_argument('--formal_epochs', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--split_num', type=int, default=9)
    parser.add_argument('--space', type=int, nargs='+', default=[20, 81])
    parser.add_argument('--mode', type=str, default='test')

    return parser.parse_args()


args = args_parse()

space = [Integer(args.space[0], args.space[1], name='t1v'),
         Integer(args.space[0], args.space[1], name='t2v'),
         Integer(args.space[0], args.space[1], name='t3v')]


def _lr_fn(epoch):
    if epoch < 4:
        lr = 0.0001 + (1 - 0.00001) / 4 * epoch
    else:
        e = epoch - 4
        es = 16
        lr = 0.5 * (1 + np.cos(np.pi * e / es))
    return lr


def set_rand_seed(seed=args.random_seed):
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


def custom_gp_minimize(func, dimensions, total_calls, n_initial_points=20, random_state=None, patience=20):

    lhs_sampler = Lhs(criterion="maximin", iterations=100)
    initial_samples = lhs_sampler.generate(dimensions, n_initial_points)

    initial_samples = np.array(initial_samples)

    initial_samples = np.unique(initial_samples, axis=0)

    while len(initial_samples) < n_initial_points:
        extra = np.random.randint([dim.lower for dim in dimensions], [dim.upper + 1 for dim in dimensions],
                                  size=(n_initial_points - len(initial_samples), len(dimensions)))
        initial_samples = np.vstack((initial_samples, extra))
        initial_samples = np.unique(initial_samples, axis=0)
    initial_samples = initial_samples[:n_initial_points]

    X = initial_samples.tolist()
    y = []
    for x in X:
        func_value = func(x)
        print(f"Initial sample x: {x}, y: {func_value} (type: {type(func_value)})")
        y.append(func_value)

    rf = SkRandomForestRegressor(n_estimators=100, random_state=random_state)

    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator=rf,
        acq_func="EI",
        random_state=random_state
    )

    optimizer.tell(X, y)

    best_y = min(y)
    no_improvement_count = 0

    switch_after = int(0.25 * total_calls)

    for i in range(total_calls):
        print(f"\nIteration {i + 1}/{total_calls}")
        print(f"aquisition function: {optimizer.acq_func}")
        if i < switch_after:
            acq_func = "EI"
        else:
            acq_func = "PI"

        optimizer.acq_func = acq_func
        next_x = optimizer.ask()
        while next_x in optimizer.Xi:
            next_x = optimizer.ask()

        y_next = func(next_x)

        optimizer.tell(next_x, y_next)

    best_index = np.argmin(optimizer.yi)
    best_x = optimizer.Xi[best_index]

    print('Best Param:', best_x)
    return best_x

def setup_bys_train_and_test(person, dataset, train_dataset, val_dataset, bys_epochs, device, classifier,
                             signalExtractor, lorenzExtractor, frequencyExtractor, loss_fn):
    @use_named_args(space)
    def bys_train_and_test(t1v, t2v, t3v):
        print('====={},{},{}====='.format(t1v, t2v, t3v))
        set_rand_seed(args.random_seed)
        for param in signalExtractor.parameters():
            param.requires_grad = False
        for param in lorenzExtractor.parameters():
            param.requires_grad = False
        for param in frequencyExtractor.parameters():
            param.requires_grad = False

        '''
        load data
        '''
        mode = 'feature_extractor'
        batch_size = 32
        train_data_size = len(train_dataset)
        val_data_size = len(val_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

        model = deepcopy(classifier)
        model.to(device)
        learning_rate = args.bys_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_step = 0
        epoch = bys_epochs

        '''
        train start
        '''
        loss_list = []
        params_list = []
        for i in range(epoch):
            train_one_epoch_loss = 0.0
            val_one_epoch_loss = 0.0
            train_acc = 0
            val_acc = 0
            val_acc = 0
            model.train()
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
                train_one_epoch_loss += loss.item()
                current_acc = (output.argmax(1).to('cpu') == targets).sum()
                train_acc += current_acc
            model.eval()
            labelList = []
            resultList = []
            for data in val_dataloader:
                valIter_batch_size = data[0].shape[0]
                val_t1 = torch.Tensor([t1v] * valIter_batch_size).long().to(device)
                val_t2 = torch.Tensor([t2v] * valIter_batch_size).long().to(device)
                val_t3 = torch.Tensor([t3v] * valIter_batch_size).long().to(device)
                singal, lorenz, frequency, targets = data
                singal_feature = signalExtractor(mode=mode, x_start=singal.to(device), t=val_t1)
                lorenz_feature = lorenzExtractor(mode=mode, x_start=lorenz.to(device), t=val_t2)
                frequency_feature = frequencyExtractor(mode=mode, x_start=frequency.to(device), t=val_t3)

                output = model(singal_feature.to(device), lorenz_feature.to(device), frequency_feature.to(device))
                for j in range(len(output)):
                    if targets[j] == 1:
                        labelList.append('1')
                    else:
                        labelList.append('0')

                    if output[j].argmax(0) == 1:
                        resultList.append('1')
                    else:
                        resultList.append('0')
                loss = loss_fn(output, targets.to(device))
                val_one_epoch_loss += loss.item()

            train_loss = train_one_epoch_loss / (train_data_size)
            print('train_loss:{}; train_acc:{}'.format(train_loss, train_acc / (train_data_size)))
            val_loss = val_one_epoch_loss / (val_data_size)
            print('val_loss:{}; val_acc:{}'.format(val_loss, val_acc / (val_data_size)))
            C_labelList = np.array(labelList).astype(int)
            C_resultList = np.array(resultList).astype(int)
            C = confusion_matrix(C_labelList, C_resultList, labels=[0, 1])
            acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
            sen = C[1][1] / (C[1][1] + C[1][0])
            precision = C[1][1] / (C[1][1] + C[0][1])
            f1 = (2 * precision * sen) / (precision + sen)

            loss_list.append(val_loss)
            current_params = [param.clone() for param in model.parameters()]
            params_list.append(current_params)

            if i == epoch - 1:
                loss_diff = loss_list[0] - loss_list[-1]
                param_diff = 0
                for previous, current in zip(params_list[0], params_list[-1]):
                    param_diff += torch.norm(current - previous).item()
                print('param_diff:{}; loss_diff:{}'.format(param_diff, loss_diff))
                if param_diff != 0:
                    ratio = loss_diff / param_diff
                    if ratio > 0:
                        transformed_ratio = math.atan(ratio)
                        print('atan_ratio:{}'.format(transformed_ratio))
                    else:
                        print('ratio: negative or zero, transformation undefined')
                        transformed_ratio = 1e-10
                else:
                    print('param_diff is zero, ratio undefined')
                    transformed_ratio = 1e-10

                result = transformed_ratio

        return -result

    return bys_train_and_test


def formal_train_and_test(t1v, t2v, t3v, dataset, train_dataset, test_dataset, formal_epochs, person, device,
                          classifier, signalExtractor, lorenzExtractor, frequencyExtractor, logPath, loss_fn):
    set_rand_seed(args.random_seed)
    for param in signalExtractor.parameters():
        param.requires_grad = False
    for param in lorenzExtractor.parameters():
        param.requires_grad = False
    for param in frequencyExtractor.parameters():
        param.requires_grad = False

    person = person.split('/')[-1]

    mode = 'feature_extractor'
    formal_train_data_size = len(train_dataset)
    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)

    '''
    Log
    '''
    if not os.path.exists(logPath + '/logs/' + person):
        os.makedirs(logPath + '/logs/' + person)
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}/{}[{}][{}][{}].txt'.format(person, t, t1v, t2v, t3v), 'a')
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
        print('-----epoch{}-----'.format(i))
        file.write('-----epoch{}-----'.format(i) + '\n')

        model.train()
        one_epoch_loss = 0.0
        train_acc = 0
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
            if train_step % 2 == 0:
                print('person:{},train_step:{},loss:{}'.format(person, train_step, loss.item()))
                file.write('person:{},train_step:{},loss:{}'.format(person, train_step, loss.item()) + '\n')

        acc = train_acc / (formal_train_data_size)
        print('train_acc:{}'.format(acc))
        if i == epoch - 1:
            save_path = './savedModels/{}/[{}][{}][{}].pth'.format(person, t1v, t2v, t3v)
            if not os.path.exists('./savedeModels/{}'.format(person)):
                os.makedirs('./savedModels/{}'.format(person))
            torch.save(model, save_path)
            print('model saved in {}'.format(save_path))
            file.write('model saved in {}'.format(save_path) + '\n')
        scheduler.step()

    file.close()

def test(t1v, t2v, t3v, person, dataset, test_dataset, device, signalExtractor, lorenzExtractor, frequencyExtractor, classifier, logPath):
    set_rand_seed(args.random_seed)

    mode = 'feature_extractor'
    test_data_size = len(test_dataset)
    batch_size = 32

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = deepcopy(classifier)
    model.to(device)
    if not os.path.exists(logPath + '/logs/' + person):
        os.makedirs(logPath + '/logs/' + person)
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    file = open(logPath + '/logs/{}/{}[{}][{}][{}].txt'.format(person, t, t1v, t2v, t3v), 'a')
    '''
    test start
    '''
    resultList = []
    labelList = []
    model.eval()
    accuracy = 0
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

def init_feature_extractors(device):
    '''
    hyper parameters
    '''
    signal_dims = 48
    signal_channels = 1
    signal_dim_mults = (1, 2, 2)

    lorenz_dims = 48
    frequency_dims = 64
    lorenz_channels = frequency_channels = 3
    lorenz_dim_mults = (1, 2, 4)
    frequency_dim_mults = (1, 2, 2, 2)

    mode = 'encoder_only'
    timesteps = 1000
    schedule_name = "linear_beta_schedule"

    '''
    init denoise models
    '''
    signalDenoise_model = _1DUnet(
        dim=signal_dims,
        channels=signal_channels,
        dim_mults=signal_dim_mults,
        mode=mode
    )
    lorenzDenoise_model = _2DUnet(
        dim=lorenz_dims,
        channels=lorenz_channels,
        dim_mults=lorenz_dim_mults,
        mode=mode
    )
    frequencyDenoise_model = _2DUnet(
        dim=frequency_dims,
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

    lorenzExtractor.load_state_dict(torch.load('./savedModels/extractor_lorenz_CPSC2018.pth', map_location=device))
    frequencyExtractor.load_state_dict(torch.load('./savedModels/extractor_frequency_CPSC2018.pth', map_location=device))
    signalExtractor.load_state_dict(torch.load('./savedModels/extractor_signal_CPSC2018.pth', map_location=device))

    return signalExtractor, lorenzExtractor, frequencyExtractor


def create_dataset(person, dataset):
    set_rand_seed()
    data_transform = {
        "train": transforms.Compose([transforms.Resize(48),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize(48),
                                   transforms.ToTensor()])}

    if args.mode == 'train':
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
        test_dataset = MyDataset(person, dataset,
                                 '../../Dataset/data_' + dataset + '_segments/personalIdx/' + person + '/test_idx_19.txt',
                                 mode='test', transform=data_transform['val'])
        return test_dataset


def process_individual(person, args, logPath, device, signalExtractor, lorenzExtractor, frequencyExtractor,
                       classifer_model, dataset, loss_fn):
    set_rand_seed(args.random_seed)
    print("person:{}".format(person))

    if args.mode == 'train':
        train_dataset, val_dataset, test_dataset = create_dataset(person, dataset)
        train_data_size = len(train_dataset)
        val_dataset_size = len(val_dataset)
        test_data_size = len(test_dataset)
        print("The length of train set: {}".format(train_data_size))
        print("The length of val set: {}".format(val_dataset_size))
        print("The length of test set: {}".format(test_data_size))

        fixed_func = setup_bys_train_and_test(person, dataset, train_dataset, val_dataset, args.bys_epochs, device,
                                              classifer_model, signalExtractor, lorenzExtractor, frequencyExtractor,
                                              loss_fn)

        result = custom_gp_minimize(
            func=fixed_func,
            dimensions=space,
            total_calls=args.bys_calls,
            n_initial_points=20,
            random_state=args.random_seed,
        )
        best_t1v, best_t2v, best_t3v = result
        print("best_t1v:{}".format(best_t1v))
        print("best_t2v:{}".format(best_t2v))
        print("best_t3v:{}".format(best_t3v))
        formal_train_and_test(best_t1v, best_t2v, best_t3v, dataset, train_dataset, test_dataset, args.formal_epochs,
                              person, device, classifer_model, signalExtractor, lorenzExtractor, frequencyExtractor,
                              logPath, loss_fn)
    elif args.mode == 'test':
        test_dataset = create_dataset(person, dataset)
        print("The length of test set: {}".format(len(test_dataset)))
        modelPath = listdir('./savedModels/{}/'.format(person))
        classifer_model = torch.load(modelPath[0], map_location=device)
        t1v, t2v, t3v = modelPath[0].split('/')[-1].strip('.pth').split('][')
        t1v = int(t1v.split('[')[1])
        t2v = int(t2v)
        t3v = int(t3v.strip(']'))
        test(t1v, t2v, t3v, person, dataset, test_dataset, device, signalExtractor, lorenzExtractor, frequencyExtractor, classifer_model, logPath)

def main():
    Name = '[mode@{}]_per_bys_[bysEpochs@{}]_[bysCalls@{}]_[bysLR@{}]_[space@{}to{}]_[formalEpochs@{}]_[formalLR@{}]'.format(
        args.mode,
        args.bys_epochs,
        args.bys_calls,
        args.bys_lr,
        args.space[0], args.space[1],
        args.formal_epochs,
        args.formal_lr,
    )

    set_rand_seed(args.random_seed)
    dataset = "LTAF"
    logPath = './per_logs/{}/{}'.format(Name, dataset)

    list = listdir('../../Dataset/data_' + dataset + '_segments/personalIdx/')
    list.sort()
    person_list = list

    split_num = args.split_num
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    person_groups = [person_list[i::split_num] for i in range(split_num)]
    signalExtractor, lorenzExtractor, frequencyExtractor = init_feature_extractors(device)
    classifier_model = CrossAttnClassifier()
    loss_fn = nn.CrossEntropyLoss()

    processes = []
    for group in person_groups:
        for person in group:
            person = person.split('/')[-1]
            p = Process(target=process_individual, args=(
                person, args, logPath, device, signalExtractor, lorenzExtractor, frequencyExtractor,
                classifier_model,
                dataset, loss_fn))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()