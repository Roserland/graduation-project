# This file is a demo of the CG-CNN-RNN-Select10

# Show a thumbnail of TCGA-WSI image locally
# Enter the patient_id, turn to the image file
# 最好是找一个病人只有一张WSI的case
# GBM 和 LGG最好都有展示

# 服务器对这些WSI进行预测
# 发回对应的显著性patch名，10张

# merge 后的图片本地存储， 实验演示

# 测试正确率？
# CG-CNN, CG-CNN-RNN



import pandas as pd
import numpy as np
import sys
import os
import argparse
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

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--valid', type=bool, default=True, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=2, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')


def main():
    global args
    args = parser.parse_args()

    # load libraries
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    demo_data = myDataset(csv_path='./coords/G_1_WSI.csv', transform=trans)
    demo_data.setmode(1)

    demo_loader = torch.utils.data.DataLoader(
        demo_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    # open output file
    fconv = open(os.path.join('./', 'demo_convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    # load model
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # for trible classification
    pre_state_dict = torch.load('./G_checkpoint_best.pth')['state_dict']
    model.load_state_dict(pre_state_dict)
    model.cuda()

    demo_probs = inference(1, demo_loader, model)
    demo_topk = group_argtopk(np.array(demo_data.slideIDX), demo_probs, 10)
    demo_maxs = group_max(np.array(demo_data.slideIDX), demo_probs, len(demo_data.targets))
    demo_pred = [1 if x >= 0.5 else 0 for x in demo_maxs]
    print("demo_pred is ", demo_pred)
    print("origin label:", demo_data.targets)

    demo_data.maketraindata(demo_topk)
    demo_data.make_demo_grid()

    demo_paths = demo_data.demo_path

    assert len(demo_paths) % 10 == 0

    for i in range(len(demo_paths) // 10):
        merge_patchs(demo_paths[i*10:i*10 + 10])

    print("****** demo finished *****\n")

    # rnn test
    rnn_demo_dset = rnndata(10, False, csv_path='./coords/G_1_WSI.csv', topk=demo_topk,
                            transform=trans)
    rnn_demo_loader = torch.utils.data.DataLoader(
                        rnn_demo_dset,
                        batch_size=10, shuffle=True,
                        num_workers=2, pin_memory=False)
    # make model
    embedder = ResNetEncoder('./G_checkpoint_best.pth')
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    rnn = rnn_single(128)
    rnn_state_dict = torch.load("./sorted_rnn_checkpoint_best.pth")["state_dict"]
    rnn.load(rnn_state_dict)
    rnn = rnn.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    val_loss, val_fpr, val_fnr = test_single(1, embedder, rnn, rnn_demo_loader, criterion)

    print("In RNN test, error is {}".format(val_fnr + val_fpr))
    print("RNN demo complete.\n")






def merge_patchs(grid_path, merge_dir = './demo_merges/'):
    assert len(grid_path) == 10

    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)

    res = np.zeros([5*512, 2*512, 3])

    for i in range(10):
        x = i % 5
        y = i // 5
        print("patch coord is {}".format((x, y)))

        temp_img_arr = np.array(Image.open(grid_path[i]))
        print("img array shape is {}".format(temp_img_arr.shape))
        res[x*512:x*512+512, y*512:y*512+512, :] = temp_img_arr

    res_img = Image.fromarray(res)

    img_name = os.path.basename(grid_path[0])[:23]
    print("case {} is merging".format(img_name))
    res_img.save(img_name, 'jpg')



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

class myDataset(data.Dataset):
    def __init__(self, csv_path='./coords/G_TwoTypes_Train.csv', transform=None):
        coords = pd.read_csv(csv_path)

        slides_path = coords['Path'].to_list()
        slides_label = coords['TypeName'].to_list()

        label_dict = {'LGG': 0, 'GBM': 1, }

        wsi_level_label = [label_dict[x] for x in slides_label]
        print("In demo labels are:", wsi_level_label)

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

        assert len(patch_level_label) == len(grid)

        self.grid = grid
        self.patch_labels = patch_level_label
        self.transform = transform
        self.mode = None
        self.targets = wsi_level_label
        self.slideIDX = slide_IDX
        # self.mult = lib['mult']
        # self.size = int(np.round(224 * lib['mult']))
        # self.level = lib['level']

    def setmode(self, mode):
        """
        1: inference, just get image
        2: train ,get image and its label
        :param self:
        :param mode:
        :return:
        """
        self.mode = mode

    def maketraindata(self, idxs):
        # prepare the (patch, patch_label)
        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in idxs]

    def make_demo_grid(self):
        paths = []
        length = len(self.t_data)

        assert length % 10 == 0

        for i in range(length):
            temp_grid = self.t_data[i][1]

            paths.append(temp_grid)

        self.demo_path = paths

    def shuffletraindata(self):
        # just shuffle
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            # slideIDX = self.patch_labels[index]
            # coord = self.grid[index]

            img = Image.open(self.grid[index])
            # img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            # if self.mult != 1:
            #     img = img.resize((224, 224), Image.BILINEAR)
            img = img.resize((224, 224), Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, grid_path, target = self.t_data[index]
            img = Image.open(grid_path)

            img = img.resize((224, 224), Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

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
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}\tERR: {}'.format(epoch + 1, args.ep, running_loss,
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


# GBM_LIST = ["TCGA-76-6286", "TCGA-76-6286", "TCGA-76-6660",
#             "TCGA-76-6191", "TCGA-06-AABW", "TCGA-06-A7TK",
#             "TCGA-02-0106", "TCGA-06-A7TL", "TCGA-74-6578",
#             "TCGA-02-0051", "TCGA-72-6280", "TCGA-06-A6S1",
#             "TCGA-06-A5U1", "TCGA-06-A5U0", "TCGA-74-6581",
#             "TCGA-06-0240", "TCGA-02-0059", "TCGA-28-5211",
#             "TCGA-28-6450", "TCGA-06-A6S0", "TCGA-74-6584",
#             "TCGA-06-0169", "TCGA-06-0678", "TCGA-06-0142",
#             "TCGA-06-0675", "TCGA-06-0681", "TCGA-74-6577"]
#
# LGG_LIST  = ["TCGA-P5-AE5Y", "TCGA-P5-AE5Y", 'TCGA-P5-A5F2',
#              'TCGA-P5-A5F1', 'TCGA-P5-A5EU', 'TCGA-VW-A7QS',
#              'TCGA-VW-A8F1', 'TCGA-P5-AE5X', 'TCGA-P5-A5F6',
#              "TCGA-P5-AE5Z", "TCGA-P5-AE5W", "TCGA-P5-AEF0",
#              "TCGA-P5-AE5T", "TCGA-P5-AEF4", "TCGA-E1-A7YQ",]

# G_Train = pd.read_csv('./coords/G_TwoTypes_Train.csv')
# G_Test  = pd.read_csv("./coords/G_TwoTypes_Test.csv")

def find_1_WSI_case(df, case_id_list, column_name = 'Case ID'):
    idx = []
    for item in df[column_name]:
        if item in case_id_list:
            idx.append(True)
        else:
            idx.append(False)
    idx = np.array(idx)
    print(idx)
    res = df[idx]

    return res

# G_1_WSI_Train = G_Train[G_Train['Case ID'] == GBM_LIST]
# G_1_WSI_Test  = G_Test[G_Train['Case ID'] == GBM_LIST]
# G_1_WSI_Train = find_1_WSI_case(G_Train, LGG_LIST)
# G_1_WSI_Test = find_1_WSI_case(G_Test, LGG_LIST)
#
# G_1_WSI = G_1_WSI_Train.append(G_1_WSI_Test)
#
# print(len(G_1_WSI_Test), len(G_1_WSI_Train), len(G_1_WSI))

# gbm_df = pd.read_csv("./coords/G_1_WSI_gbm.csv")
# lgg_df = pd.read_csv("./coords/G_1_WSI_lgg.csv")
# final_df = gbm_df.append(lgg_df)
#
#
# final_df.to_csv('./coords/demo_1.csv', index=None)



