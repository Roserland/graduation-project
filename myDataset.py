import sys
import os
import pandas as pd
import numpy as np
import random
import PIL.Image as Image
import torch
import json
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.models as models

def addPrefix(prefix, fileName):
    return os.path.join(prefix, fileName)

class myDataset(data.Dataset):
    def __init__(self, csv_path='./coords/threeTypes_train.csv', transform=None):
        coords = pd.read_csv(csv_path)

        slides_path = coords['Path'].to_list()
        slides_label = coords['TypeName'].to_list()

        label_dict = {'LUSC': 0, 'LUAD': 1, }

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

class myDataset_2(data.Dataset):
    def __init__(self, csv_path='./coords/threeTypes_train.csv', transform=None):
        coords = pd.read_csv(csv_path)

        slides_path = coords['Path'].to_list()
        slides_label = coords['TypeName'].to_list()

        label_dict = {'UCEC': 0, 'PAAD': 1, 'CESC': 2}

        patch_level_label = []
        grid = []
        for i, wsi_direc in enumerate(slides_path):
            patch_name_list = os.listdir(wsi_direc)  # g is a slide directory path

            # add prefix, so the patch has its full path
            temp_full_path = [os.path.join(wsi_direc, x) for x in patch_name_list]

            grid.extend(temp_full_path)
            # patch label is WSI label
            temp_label = label_dict[slides_label[i]]
            patch_level_label.extend([temp_label] * len(patch_name_list))
        print('Number of tiles: {}'.format(len(grid)))

        assert len(patch_level_label) == len(grid)

        self.grid = grid
        self.patch_labels = patch_level_label
        self.transform = transform
        self.mode = None
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
        self.t_data = [(self.grid[x], self.patch_labels[x]) for x in idxs]

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
            # slideIDX, coord, target = self.t_data[index]
            img = Image.open(self.grid[index])
            target = self.patch_labels[index]
            # img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            # if self.mult != 1:
            #     img = img.resize((224, 224), Image.BILINEAR)
            img = img.resize((224, 224), Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

class randSet(data.Dataset):
    # for each WSI, random choose 100 tiles
    # each tile label is the WSI-Level label
    def __init__(self, csv_path='./coords/KIPAN_ThreeTypes_Test.csv', sampleNum = 100, transform=None):
        coords = pd.read_csv(csv_path)

        slides_path = coords['Path'].to_list()
        slides_label = coords['TypeName'].to_list()

        label_dict = {'KICH': 0, 'KIRC': 1, 'KIRP': 2}

        patch_level_label = []
        grid = []
        wsi_indexs = [0]
        curr_len = 0
        for i, wsi_direc in enumerate(slides_path):
            patch_name_list = os.listdir(wsi_direc)  # g is a slide directory path

            # sampling
            # print(len(patch_name_list), sampleNum)
            if len(patch_name_list) < sampleNum:
                snum = len(patch_name_list)
            else:
                snum = sampleNum
            patch_name_list = random.sample(patch_name_list, snum)

            # add prefix, so the patch has its full path
            temp_full_path = [os.path.join(wsi_direc, x) for x in patch_name_list]

            # prepare WSI index, each number is the begin of a wsi
            curr_len += snum
            wsi_indexs.append(curr_len)

            grid.extend(temp_full_path)
            # patch label is WSI label
            temp_label = label_dict[slides_label[i]]
            patch_level_label.extend([temp_label] * len(patch_name_list))

        assert len(patch_level_label) == len(grid)
        assert len(wsi_indexs) == len(slides_path) + 1
        assert len(grid) == wsi_indexs[-1]
        print('Number of tiles: {}'.format(len(grid)))

        num_wsi_label = [label_dict[x] for x in slides_label]

        self.grid = grid
        self.patch_labels = patch_level_label
        self.wsi_indexs = wsi_indexs
        self.wsi_label  = num_wsi_label
        self.transform = transform
        self.mode = None

    def setmode(self, mode):
        """
        1: inference, just get image
        2: train ,get image and its label
        :param self:
        :param mode:
        :return:
        """
        self.mode = mode


    def __getitem__(self, index):
        img = Image.open(self.grid[index])
        target = self.patch_labels[index]
        img = img.resize((224, 224), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.grid)

if __name__ == '__main__':
    csv_path = './coords/threeTypes_train.csv'

    coords = pd.read_csv(csv_path)

    slides_path = coords['Path'].to_list()
    slides_label = coords['TypeName'].to_list()

    grid = []
    patch_level_label = []

    for i, wsi_direc in enumerate(slides_path):
        patch_name_list = os.listdir(wsi_direc)  # g is a slide directory path

        # add prefix, so the patch has its full path
        temp_full_path = [os.path.join(wsi_direc, x) for x in patch_name_list]

        grid.extend(temp_full_path)
        # patch label is WSI label
        patch_level_label.extend(slides_label[i] * len(patch_name_list))

    len_grid = len(grid)
    rand = np.random.randint(0, len_grid)

    train_set = myDataset()
    train_set.setmode(1)

    img_arr = np.array(train_set[rand])
    print(img_arr[:, :20, 1])

    train_set[rand].save('./temp.jpg')

    print('finished')



