from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lib import FLAGS
from lib.GeneralController import GeneralController
from ipdb import set_trace

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import torchvision
import torch
import torch.optim as optim

from tqdm import tqdm

import numpy as np

def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    path = "./data"

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    trn_X = np.zeros(shape=(len(trainloader), FLAGS.INPUT_CHANNELS, FLAGS.R, FLAGS.C))
    trn_y = np.zeros(shape=(len(trainloader)), dtype="long")
    for i, (inputs, targets) in enumerate(tqdm(trainloader)):
        trn_X[i] = inputs.numpy()[0]
        trn_y[i] = targets.numpy()[0]
        if i > len(trainloader)//10:
            break

    val_X = np.zeros(shape=(len(testloader), FLAGS.INPUT_CHANNELS, FLAGS.R, FLAGS.C))
    val_y = np.zeros(shape=(len(testloader)), dtype="long")
    for i, (inputs, targets) in enumerate(tqdm(testloader)):
        val_X[i] = inputs.numpy()[0]
        val_y[i] = targets.numpy()[0]
        if i > len(testloader)//10:
            break

    trn = [trn_X, trn_y]
    val = [val_X, val_y]

    data = ImageClassifierData.from_arrays(path, trn=trn, val=val,
                                           classes=classes)

    controller = GeneralController()
    arch, sample_arc = controller()

    learn = Learner(data=data, models=arch)
    learn.crit = arch.crit
    learn.model.train()

    lrf = learn.lr_find()
    print(lrf)

    # if FLAGS.TRAIN:
    #     train(controller, train_loader)
    # else:
    #     test(controller, test_loader)


if __name__ == "__main__":
    main()
