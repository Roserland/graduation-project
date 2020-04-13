import sys
import os
import pandas as pd
import numpy as np
import random
# import openslide
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
        slides = []
        # for i, name in enumerate(lib['slides']):
        #     sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
        #     sys.stdout.flush()
        #     slides.append(openslide.OpenSlide(name))

        # Flatten grid
        # prepare the patches, concentrate all patches into a list

        # slideIDX = []
        patch_level_label = []
        grid = []
        for i, wsi_direc in enumerate(slides_path):
            patch_name_list = os.listdir(wsi_direc)  # g is a slide directory path

            # add prefix, so the patch has its full path
            temp_full_path = [os.path.join(wsi_direc, x) for x in patch_name_list]

            grid.extend(temp_full_path)
            # patch label is WSI label
            patch_level_label.extend([slides_label[i]] * len(patch_name_list))
        print('Number of tiles: {}'.format(len(grid)))
        print('Number of label: {}'.format(len(patch_level_label)))

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
            slideIDX = self.patch_labels[index]
            coord = self.grid[index]

            img = Image.open(grid[index])
            # img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            # if self.mult != 1:
            #     img = img.resize((224, 224), Image.BILINEAR)
            img = img.resize((224, 224), Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img, target = self.patch_labels[index]
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

    print(train_set[rand])

    img = Image.open(train_set[rand])
    print(img[:, :20, 1])

    img.save('./temp.jpg')

    print('finished')



