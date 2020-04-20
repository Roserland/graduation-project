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
# ids = [0,1]

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--valid', type=bool, default=True, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=4, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=100, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

best_acc = 0

def main():

    global args, best_acc
    args = parser.parse_args()

    # resnet-34, or could change the model for efficiency
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)           # for trible classification
    pre_state_dict = torch.load('./checkpoints/LU_V2.pth')['state_dict']
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
    train_dset = myDataset(csv_path='./coords/LU_TwoTypes_Train.csv', transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.valid:
        val_dset = myDataset(csv_path='./coords/LU_TwoTypes_Test.csv', transform=trans)
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
        # for evaluation,
        train_dset.setmode(1)

        # slideIDX --> Patch_level label

        # get all problities of all patches in the loader
        probs = inference(epoch, train_loader, model)
        print(probs.shape)

        # choss top-k patches with high probilities to train
        # topk is an index
        topk = group_argtopk(np.array(train_dset.patch_labels), probs, args.k)
        # make train data, and shuffle it
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()

        # training part
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        logger.info('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1, loss))
        fconv.close()
        
        torch.save(model.state_dict(), os.path.join(args.output, 'LU_current_checkpoint.pth'))

        # Validation
        if (epoch + 1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.patch_labels), probs, len(val_dset.patch_labels))
            logger.info('In validation, (most 128) predicted probs are', pred)
            logger.info(maxs[:128])
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            # logger.info('In validation, predicted probs are', pred)
            # print(pred)
            logger.info(pred)
            err, fpr, fnr = calc_err(pred, val_dset.patch_labels)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch + 1, args.nepochs, err, fpr,
                                                                                   fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch + 1, err))
            fconv.write('{},fpr,{}\n'.format(epoch + 1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch + 1, fnr))
            fconv.close()
            # Save best model
            err = (fpr + fnr) / 2.
            if 1 - err >= best_acc:
                best_acc = 1 - err
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output, 'LU_checkpoint_best.pth'))


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
