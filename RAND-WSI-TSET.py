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
from collections import Counter
import logging

from myDataset import randSet

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--valid', type=bool, default=True, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=5, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=10, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument('--sample', default=100, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')


def main():
    acc = 0

    global args
    args = parser.parse_args()

    model = models.resnet50(False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # for trible classification
    weightPath = './checkpoint_best.pth'
    state_dict = torch.load(weightPath)['state_dict']
    model.load_state_dict(state_dict)

    model.cuda()

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    val_dset = randSet(csv_path='./coords/threeTypes_test.csv', sampleNum=args.sample, transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # How to aggregate to WSI model

    val_dset.setmode(2)
    model.eval()
    patch_pred = []
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            # print('Test\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch + 1, args.nepochs, i + 1, len(val_loader)))
            input = input.cuda()
            pred = model(input).cpu().numpy()
            pred = np.argmax(pred, axis=1).tolist()

            patch_pred.extend(pred)
            # acc_num += (pred == np.array(label)).sum()

    # calculate the WSI level prediction accuracy
    # using voting strategy
    wsi_pred = []
    for i in range(0, len(val_dset.wsi_indexs) - 1):
        begin = val_dset.wsi_indexs[i]
        end   = val_dset.wsi_indexs[i+1]

        patch_labels = val_dset.patch_labels[begin:end]

        # voting
        cnt = Counter(patch_labels)
        most_common_key = cnt.most_common(1)[0][0]

        wsi_pred.append(most_common_key)
    print(wsi_pred)

    pred_arr = np.array(wsi_pred)
    orig_arr = np.array(val_dset.wsi_label)
    assert len(orig_arr) == len(pred_arr)
    print(pred_arr)
    print(orig_arr)

    wsi_acc = (pred_arr == orig_arr).sum() / len(pred_arr)
    # wsi_acc = (np.array(wsi_pred) == np.array(val_dset.wsi_label)).astype(int).sum() / len(wsi_pred)
    print("WSI level accuracy is {}".format(wsi_acc))

if __name__ == '__main__':
    main()