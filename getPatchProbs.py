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

from myDataset import myDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
best_acc = 0

parser = argparse.ArgumentParser(description='ABB')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=2, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')


def main():
    global args, best_acc

    args = parser.parse_args()

    # resnet-34, or could change the model for efficiency
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # for trible classification
    pre_state_dict = torch.load('./G_checkpoint_best.pth')['state_dict']
    # #pre_state_dict = torch.load('./checkpoints/LU_V3.pth')
    model.load_state_dict(pre_state_dict)
    model.cuda()

    device_ids = range(torch.cuda.device_count())
    print(device_ids)

    # if necessary, mult-gpu training
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load data
    train_dset = myDataset(csv_path='./coords/G_TwoTypes_Train.csv', transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    val_dset = myDataset(csv_path='./coords/G_TwoTypes_Test.csv', transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)



    # loop throuh epochs
    # for evaluation,
    train_dset.setmode(1)
    val_dset.setmode(1)
    # slideIDX --> Patch_level label
    # get all problities of all patches in the loader
    train_probs = inference(123, train_loader, model)
    print(train_probs.shape)
    val_probs = inference(123, val_loader, model)
    # choose most K probable patch per slide, k = 2
    train_topk = group_argtopk(np.array(train_dset.slideIDX), train_probs, 10)
    val_topk   = group_argtopk(np.array(val_dset.slideIDX), val_probs, 10)
    print(len(train_dset.slideIDX), len(train_topk))
    print(len(val_dset.slideIDX), len(val_topk))

    data = {
        "train_patch_probs": train_probs,
        "val_patch_probs":val_probs,
        "train_top_k":train_topk,
        "val_top_k":val_topk
    }
    torch.save(data, './for_rnn_train.pth')

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