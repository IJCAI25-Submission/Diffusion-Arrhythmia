import torch
import torch.nn as nn
from Baselines.Models.UNet_1D import Unet as UNet1D
from Baselines.Models.UNet_2D import Unet as UNet2D

class BaseVGG11_2D(nn.Module):
    def __init__(self):
        super(BaseVGG11_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Dropout(0.5),
            # nn.Linear(2048, 2),
        )
        self.fc = nn.Linear(2048, 128)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)

        return x

class BaseVGG11_1D(nn.Module):
    def __init__(self):
        super(BaseVGG11_1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(79872, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Dropout(0.5),
            # nn.Linear(2048, 2),
        )
        self.fc = nn.Linear(2048, 128)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)

        return x

class BaseUnetCNN_1D(nn.Module):
    def __init__(self, view='signal'):
        super(BaseUnetCNN_1D, self).__init__()
        self.unet = UNet1D(dim=48, dim_mults=(1, 2, 2), mode='encoder_only')
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(2880, 512)
        self.fc1 = nn.Linear(60000, 128)

    def forward(self, x):
        x = self.unet(x)
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        x = self.fc1(x)

        return x

class BaseUnetCNN_2D(nn.Module):
    def __init__(self, view='lorenz'):
        super(BaseUnetCNN_2D, self).__init__()
        self.view = view
        if self.view == 'lorenz':
            self.unet = UNet2D(dim=48, dim_mults=(1, 2, 4), mode='encoder_only')
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # self.fc1 = nn.Linear(2880, 512)
            self.fc1 = nn.Linear(27648, 128)
        elif self.view == 'frequency':
            self.unet = UNet2D(dim=64, dim_mults=(1, 2, 2, 2), mode='encoder_only')
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # self.fc1 = nn.Linear(2880, 512)
            self.fc1 = nn.Linear(4608, 128)

    def forward(self, x):
        x = self.unet(x)
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        x = self.fc1(x)

        return x

class MoCo2D(nn.Module):
    def __init__(self, device='cpu', dim=128, K=8192, m=0.99, T=0.07, base_model=None, view=None):
        super(MoCo2D, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.view = view

        if base_model == 'unet':
            self.encoder_q = BaseUnetCNN_2D(self.view)
            self.encoder_k = BaseUnetCNN_2D(self.view)
        elif base_model == 'vgg11':
            self.encoder_q = BaseVGG11_2D()
            self.encoder_k = BaseVGG11_2D()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.mul_(self.m).add_(param_q * (1. - self.m))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        self._dequeue_and_enqueue(k)

        return logits, labels

class MoCo1D(nn.Module):
    def __init__(self, device='cpu', base_model='cnn', dim=128, K=8192, m=0.99, T=0.07, view=None):
        super(MoCo1D, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.view = view

        if base_model == 'vgg11':
            self.encoder_q = BaseVGG11_1D()
            self.encoder_k = BaseVGG11_1D()
        elif base_model == 'unet':
            self.encoder_q = BaseUnetCNN_1D(self.view)
            self.encoder_k = BaseUnetCNN_1D(self.view)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.mul_(self.m).add_(param_q * (1. - self.m))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        self._dequeue_and_enqueue(k)

        return logits, labels

class multiChannelDownStreamClassifier(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, num_classes):
        super(multiChannelDownStreamClassifier, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.fc = nn.Linear(128, num_classes)

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(128)

    def forward(self, signal, lorenz, frequency):
        x1 = self.encoder1(signal)
        x2 = self.encoder2(frequency)
        x3 = self.encoder3(lorenz)
        modalities = torch.stack([x1, x2, x3], dim=1)
        attn_output, _ = self.attention(modalities, modalities, modalities)
        attn_output = self.attention_norm(attn_output)
        fused_features = torch.mean(attn_output, dim=1)

        x = self.fc(fused_features)

        return x

# i = torch.randn(64, 3, 48, 48)
# j = torch.randn(64, 3, 48, 48)
# k = torch.randn(64, 1, 2500)
# base1 = MoCo1D(device='cpu', base_model='unet', view='signal')
# base2 = MoCo2D(device='cpu', base_model='unet', view='frequency')
# base3 = MoCo2D(device='cpu', base_model='unet', view='lorenz')
# model = multiChannelDownStreamClassifier(base1.encoder_q, base2.encoder_q, base3.encoder_q, 2)
# o = model(k, i, j)
# print(o.shape)