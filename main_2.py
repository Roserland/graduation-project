# This file test the random selected tiles to train a
# patch-level model f1
# Then use f1 to predict WSI label.

import sys
import os
import argparse
import numpy as np
import random
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import logging

from myDataset import randSet

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--valid', type=bool, default=True, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=5, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=10, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument('--sample', default=100, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log_main2.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():

    global args, best_acc
    args = parser.parse_args()

    # resnet-34, or could change the model for efficiency
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 3)           # for trible classification
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load data
    train_dset = randSet(csv_path='./coords/threeTypes_train.csv', sampleNum=args.sample, transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    if args.valid:
        val_dset = randSet(csv_path='./coords/threeTypes_test.csv', sampleNum=args.sample, transform=trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    # open output file
    fconv = open(os.path.join(args.output, 'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):

        # training part
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        logger.info('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        # Validation
        if args.val_lib and (epoch + 1) % args.test_every == 0:
            val_dset.setmode(2)
            model.eval()
            acc_num = 0
            with torch.no_grad():
                for i, (input, label) in enumerate(val_loader):
                    print('Test\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch + 1, args.nepochs, i + 1, len(val_loader)))
                    input = input.cuda()
                    pred = model(input).numpy()
                    pred = np.argmax(pred, axis=1)

                    acc_num += (pred == np.array(label)).sum()
            acc = acc_num / len(val_dset)

            print('Validation\tEpoch: [{}/{}]\tACC: {}'.format(epoch + 1, args.nepochs, acc))

            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},acc,{}\n'.format(epoch + 1, acc))
            fconv.close()

            if acc > best_acc:
                best_acc = acc
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output, 'checkpoint_best.pth'))
                torch.save(optimizer.state_dict(), 'weight.pth')


def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def test(run, loader, model, criterion, optimizer):
    model.eval()
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Test\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run + 1, args.nepochs, i + 1, len(loader)))
            input = input.cuda()
            output = model(input)


def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


if __name__ == '__main__':
    main()
