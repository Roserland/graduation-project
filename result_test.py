# This file is a demo of the CG-CNN-RNN-Select10

# Show a thumbnail of TCGA-WSI image locally
# Enter the patient_id, turn to the image file
# 最好是找一个病人只有一张WSI的case
# GBM 和 LGG最好都有展示

# 服务器对这些WSI进行预测
# 发回对应的显著性patch名，10张

# merge 后的图片本地存储， 实验演示


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

def main():
    demo_data = myDataset(csv_path='./coords/G_1_WSI.csv')

    demo_loader = torch.utils.data.DataLoader(
        demo_data,
        batch_size=10, shuffle=False,
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

    demo_data.maketraindata(demo_topk)

    demo_paths = demo_data.demo_path

    assert len(demo_paths) % 10 == 0

    for i in range(len(demo_paths) // 10):
        merge_patchs(demo_paths[i*10:i*10 + 10])

    print("demo finished")



def merge_patchs(grid_path, merge_dir = './demo_merges/'):
    assert len(grid_path) == 10

    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)

    res = np.zeros([5*512, 2*512, 3])

    for i in range(10):
        x = i % 5
        y = i - x
        print(x, y)

        temp_img_arr = np.array(Image.open(grid_path[i]))
        res[x, y, :] = temp_img_arr

    res_img = Image.fromarray(res)

    img_name = os.path.basename(grid_path[0])

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



