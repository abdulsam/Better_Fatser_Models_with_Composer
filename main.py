import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

# For Summary of Model
from ptflops import get_model_complexity_info
from pytorch_model_summary import summary


# WandB – Import the wandb library
import wandb

# MosaicML Composer
from composer import functional as CF
from composer.algorithms.randaugment import RandAugmentTransform 


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with Composer Library functions and wandb')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# Data
print('--> Augmenting data..')

#--- Composer Rand Aug ---#
randaugment_transform = RandAugmentTransform(severity=9,
                                             depth=2,
                                             augmentation_set="all")
transform_train_data = transforms.Compose([
    randaugment_transform,
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train_data)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test_data)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('-- ResNet-152 --')
net = ResNet152()

# -- Loss --
criterion = nn.CrossEntropyLoss()

# -- Optimizer --
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

# -- Scheduling Learning Rate --
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#--- Composer BlurPool ---#
CF.apply_blurpool(
        net,
        optimizers=optimizer,
        replace_convs=True,
        replace_maxpools=True,
        blur_first=True
    )

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# -- Summary of Model --
print(summary(net, torch.zeros((1, 3, 32, 32))))
macs, params = get_model_complexity_info(net, (3, 32, 32),
                                             as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# Training and Testing
def train_and_test(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # -- Composer MixUp Batch Data -- 
        X_mixed, y_perm, mixing = CF.mixup_batch(inputs, targets, alpha=0.2)
        outputs = net(X_mixed)

        # -- Composer Label Smoothing --
        smoothed_targets = CF.smooth_labels(outputs, targets, smoothing=0.1)

        # -- Loss Calculation -- because of mixing and label smoothing
        loss = (1 - mixing) * criterion(outputs, smoothed_targets) + mixing * criterion(outputs, y_perm)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    #Test script
    global best_acc
    net.eval()
    test_loss = 0
    correctt = 0
    totalt = 0
    with torch.no_grad():
        for batch_idxt, (inputst, targetst) in enumerate(testloader):
            inputst, targetst = inputst.to(device), targetst.to(device)
            outputst = net(inputst)
            losst = criterion(outputst, targetst)
            test_loss += losst.item()
            _, predictedt = outputst.max(1)
            totalt += targetst.size(0)
            correctt += predictedt.eq(targetst).sum().item()

            progress_bar(batch_idxt, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idxt+1), 100.*correctt/totalt, correctt, totalt))

    # -- Logging Info on WANDB --
    wandb.log({
        "Train Accuracy": 100. * correct / len(trainloader.dataset),
        "Train Loss": train_loss/(batch_idx+1),
        "Test Accuracy": 100. * correctt / len(testloader.dataset),
        "Test Loss": test_loss/(batch_idxt+1)})

    # Save checkpoint.
    acc = 100.*correctt/totalt
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train_and_test(epoch)
    scheduler.step()

wandb.watch(net, log="all")
# WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
torch.save(net.state_dict(), "net.h5")
wandb.save('net.h5')
wandb.finish()