import os
import sys
from PIL import Image
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


parser = argparse.ArgumentParser(description='graduation project')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, help='path to trained model checkpoint')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence')

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()

    # load libraries
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    mm = torch.load('./for_rnn_train.pth')
    topk_train = mm['train_top_k']
    topk_val   = mm['val_top_k']

    train_dset = rnndata(args.s, True, csv_path='./coords/G_TwoTypes_Train.csv', topk=topk_train,
                         transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(args.s, True, csv_path='./coords/G_TwoTypes_Test.csv', topk=topk_val,
                       transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # make model
    embedder = ResNetEncoder('./G_checkpoint_best.pth')
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    rnn = rnn_single(args.ndims)
    rnn = rnn.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    cudnn.benchmark = True

    torch.save(rnn.state_dict(), os.path.join(args.output, 'rnn_G_current_checkpoint.pth'))
    fconv = open(os.path.join(args.output, 'rnn_convergence.csv'), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,train.err,val.loss,val.fpr,val.fnr,val.err\n')
    fconv.close()

    for epoch in range(args.nepochs):

        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
        val_loss, val_fpr, val_fnr = test_single(epoch, embedder, rnn, val_loader, criterion)
        train_err = (train_fpr + train_fnr) / 2
        val_err = (val_fpr + val_fnr) / 2
        fconv = open(os.path.join(args.output, 'sorted_rnn_convergence.csv'), 'a')
        fconv.write(
            '{},{},{},{},{},{},{},{},{}\n'.format(epoch + 1, train_loss, train_fpr, train_fnr, train_err,
                                            val_loss, val_fpr, val_fnr, val_err))
        fconv.close()

        val_err = (val_fpr + val_fnr) / 2
        print(val_err)
        if 1 - val_err >= best_acc:
            best_acc = 1 - val_err
            obj = {
                'epoch': epoch + 1,
                'state_dict': rnn.state_dict()
            }
            torch.save(obj, os.path.join(args.output, 'sorted_rnn_checkpoint_best.pth'))


def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i, (inputs, target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch + 1, args.nepochs, i + 1, len(loader)))

        batch_size = inputs[0].size(0)
        rnn.zero_grad()

        state = rnn.init_hidden(batch_size).cuda()
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            _, input = embedder(input)
            output, state = rnn(input, state)

        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss / len(loader.dataset)
    running_fps = running_fps / (np.array(loader.dataset.targets) == 0).sum()
    running_fns = running_fns / (np.array(loader.dataset.targets) == 1).sum()
    running_err = (running_fns + running_fps) / 2
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}\tERR: {}'.format(epoch + 1, args.nepochs, running_loss,
                                                                         running_fps, running_fns, running_err))
    return running_loss, running_fps, running_fns


def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch + 1, args.nepochs, i + 1, len(loader)))

            batch_size = inputs[0].size(0)

            state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)

            target = target.cuda()
            loss = criterion(output, target)

            running_loss += loss.item() * target.size(0)
            fps, fns = errors(output.detach(), target.cpu())
            running_fps += fps
            running_fns += fns

    running_loss = running_loss / len(loader.dataset)
    running_fps = running_fps / (np.array(loader.dataset.targets) == 0).sum()
    running_fns = running_fns / (np.array(loader.dataset.targets) == 1).sum()
    running_err = (running_fns + running_fps) / 2

    print('Validating - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}\tERR: {}'.format(epoch + 1, args.nepochs, running_loss,
                                                                           running_fps, running_fns, running_err))
    return running_loss, running_fps, running_fns


def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred != real
    fps = float(np.logical_and(pred == 1, neq).sum())
    fns = float(np.logical_and(pred == 0, neq).sum())
    return fps, fns


class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()

        temp = models.resnet34()
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(path)
        temp.load_state_dict(ch['state_dict'])
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x), x


class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(512, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state + input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)

class rnndata(data.Dataset):
    def __init__(self, s, shuffle=False, csv_path='./coords/G_TwoTypes_Train.csv', topk = [],
                 transform=None):
        coords = pd.read_csv(csv_path)

        assert len(topk) == 10*len(coords)

        slides_path = coords['Path'].to_list()
        slides_label = coords['TypeName'].to_list()

        label_dict = {'LGG': 0, 'GBM': 1, }

        wsi_level_label = [label_dict[x] for x in slides_label]

        patch_level_label = []
        grid = []
        slide_IDX = []
        for i, wsi_direc in enumerate(slides_path):
            patch_name_list = os.listdir(wsi_direc)  # g is a slide directory path

            # add prefix, so the patch has its full path
            temp_full_path = [os.path.join(wsi_direc, x) for x in patch_name_list]

            grid.extend(temp_full_path)
            # patch label is WSI label
            temp_label = label_dict[slides_label[i]]
            patch_level_label.extend([temp_label] * len(patch_name_list))

            slide_IDX.extend([i] * len(patch_name_list))

        print('Number of tiles: {}'.format(len(grid)))

        self.slide_names = slides_path
        self.targets = wsi_level_label
        self.s = s
        self.shuffle = shuffle
        self.transform = transform
        self.grid = grid
        self.patch_labels = patch_level_label
        self.top_k = topk

        print('loading finish, {} WSIs have been loaded'.format(len(self.targets)))

    def __getitem__(self, index):
        single_silde_path = self.slide_names[index]

        grid_idx = self.top_k[index*10:index*10 + 10]

        temp_grid = [self.grid[idx] for idx in grid_idx]

        patch_name_list = os.listdir(single_silde_path)
        # add prefix, so the patch has its full path
        grid = [os.path.join(single_silde_path, x) for x in patch_name_list]

        if self.shuffle:
            temp_grid = random.sample(temp_grid, len(temp_grid))

        out = []
        s = min(self.s, len(grid))
        for i in range(s):
            img = Image.open(temp_grid[i])
            img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)
        return out, self.targets[index]


    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    main()
    # mm = torch.load('./for_rnn_train.pth')
    #     # print(mm.keys())
    #     #
    #     # print(len(mm['train_top_k']))
    #     # print(mm['train_top_k'])