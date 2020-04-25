import sys
import os
import argparse
import pandas as pd
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ids = [0,1]

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--valid', type=bool, default=True, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
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
    #pre_state_dict = torch.load('./checkpoints/LU_V3.pth')
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

    val_dset = myDataset(csv_path='./coords/G_TwoTypes_Test.csv', transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    val_dset.setmode(1)

    getMostProbPatchs(val_loader, val_dset, model, csv_name = './selected_test.csv')


    train_dset = myDataset(csv_path='./coords/G_TwoTypes_Train.csv', transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    train_dset.setmode(1)
    getMostProbPatchs(train_loader, train_dset, model, csv_name = './selected_train.csv')


def getMostProbPatchs(train_loader, train_dset, model, csv_name = './selected_train.csv'):
    probs_train = inference(1, train_loader, model)
    print(probs_train.shape)
    # choose most K probable patch per slide, k = 2
    topk_train = group_argtopk(np.array(train_dset.slideIDX), probs_train, args.k)

    patch_list = np.array(train_dset.grid)
    topk_train = np.array(topk_train)

    selected = patch_list[topk_train].tolist()

    df = {'selected_patch_path': selected}
    df = pd.DataFrame(df)
    df.to_csv(csv_name)

    print("selceted patchs num: {}".format(len(df)))

    for item in selected:
        basename = os.path.basename(item)

        new_path = os.path.join('./selected_patchs', basename)

        img = Image.open(item)

        print("save path:", new_path)

        img.save(new_path)








def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run + 1, args.nepochs, i + 1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i * args.batch_size:i * args.batch_size + input.size(0)] = output.detach()[:, 1].clone()
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
        running_loss += loss.item() * input.size(0)
    return running_loss / len(loader.dataset)


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    return err, fpr, fnr


def group_argtopk(groups, data, k=1):
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