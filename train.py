import os
import numpy as np

from dataset import BirdsDataset
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
from torchvision.models import resnet34
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from apex import amp

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.
    for mini_batch in tqdm(loader, desc="Training"):
        imgs = mini_batch['image'].cuda()
        targets = mini_batch['label'].cuda()

        optimizer.zero_grad()
        logits = model(imgs)

        loss = criterion(logits, targets)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(loader)
    return train_loss


def validate(model, loader, criterion, total_set_size):
    model.eval()
    running_loss = 0.
    running_correct = 0

    for mini_batch in tqdm(loader, desc="Validating"):
        imgs = mini_batch['image'].cuda()
        targets = mini_batch['label'].cuda()

        with torch.no_grad():
            logits = model(imgs)
            loss = criterion(logits, targets)
            running_loss += loss.item()

            # softmax across logits
            preds = torch.nn.functional.softmax(logits, dim=1)
            # argmax to get class prediction
            preds = preds.argmax(dim=1)

            # add correct preds to running total
            running_correct += (preds == targets).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = float(running_correct / total_set_size)
    return val_loss, val_acc


if __name__ == '__main__':
    torch.manual_seed(0)
    DATA_DIR = '/home/ali/Documents/Datasets/225_bird_species/consolidated'
    VAL_SPLIT = 0.2
    EPOCHS = 15
    BATCH_SIZE = 128

    train_transforms = Compose([RandomHorizontalFlip(),
                                ColorJitter(0.6, 0.3, 0.1, 0.1),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                                ])
    val_transforms = Compose([ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                              ])

    # make lazy dataset objects from master so different transforms can be used for train/val
    # these datasets will be filtered down to only include their respective samples with Subset()
    dataset = ImageFolder(DATA_DIR)
    train_dataset = BirdsDataset(dataset, train_transforms)
    val_dataset = BirdsDataset(dataset, val_transforms)

    # stratified split
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=VAL_SPLIT, stratify=dataset.targets)
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)

    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4,
                            drop_last=True, pin_memory=True, shuffle=False)

    # MAKE NETWORK
    net = resnet34(pretrained=True, progress=True)
    net.fc = torch.nn.Linear(net.fc.in_features, len(dataset.classes))
    net.cuda()
    # MAKE OPTIMIZER
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # MAKE CRITERION
    loss_fn = torch.nn.CrossEntropyLoss()  # takes raw logits
    # MAKE SCHEDULER
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     factor=0.1,
                                     patience=2,
                                     mode='min',
                                     verbose=True)
    # Auto Mixed Precision
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    # INITIALIZE TENSORBOARD
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    writer = SummaryWriter(f"./logs/{dt_string}")

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        train_loss = train(net, train_loader, loss_fn, optimizer)
        writer.add_scalar('loss/train', train_loss, epoch)
        val_loss, val_acc = validate(net, val_loader, loss_fn, len(val_dataset))
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('acc/classification_acc', val_acc, epoch)

        writer.add_scalar('lr/train', optimizer.param_groups[0]['lr'], epoch)
        lr_scheduler.step(val_loss)
        writer.flush()

    writer.close()

