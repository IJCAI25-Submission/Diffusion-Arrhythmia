import sys
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from moco_load_data import MoCoDataset2D
from Baselines.Models.MultiMoCoCreator import MoCo2D as MoCo
import numpy as np
import random
import argparse
def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--formal_epochs', type=int, default=100)
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='CPSC2018')
    parser.add_argument('--modality', type=str, default='lorenz')
    parser.add_argument('--baseModel', type=str, default='unet')
    return parser.parse_args()

args = args_parse()

def set_rand_seed(seed=args.random_seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_rand_seed()
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

def train_moco(moco, train_loader, optimizer, criterion, device):
    moco.train()
    total_loss = 0.0
    for i, (x_i, x_j) in enumerate(train_loader):
        x_i, x_j = x_i.to(device), x_j.to(device)
        logits, labels = moco(x_i, x_j)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moco.parameters(), max_norm=2.0)
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print('train_step:{},loss:{}'.format(i, loss.item()))
    return total_loss / len(train_loader)

dataset = args.dataset
batch_size = 64
epochs = args.formal_epochs
lr = args.formal_lr
temperature = args.temperature

train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])
train_dataset = MoCoDataset2D(dataset, '../../Dataset/data_'+dataset+'_segments/person_all_idx.txt', args.modality, train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

moco = MoCo(device=device, base_model=args.baseModel, view=args.modality).to(device)

optimizer = optim.SGD(moco.encoder_q.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min=0, last_epoch=-1)
criterion = nn.CrossEntropyLoss()

best_loss = 1000

for epoch in range(epochs):
    loss = train_moco(moco, train_loader, optimizer, criterion, args.cuda)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss}')
    scheduler.step()
    if loss < best_loss:
        best_loss = loss
        torch.save(moco, './MoCoModel_{}_{}_{}.pth'.format(args.dataset, args.baseModel, args.modality))
        print('The model has been saved\n')