import sys
import os
import pandas as pd
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
from sklearn.svm import SVC
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Use ResNet-34 to extract features
#

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--valid', type=bool, default=True, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='./', help='name of output file')
parser.add_argument('--batch_size', type=int, default=1024, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=2, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=200, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./CNN-FEA-SVM_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class res34(nn.Module):
    def __init__(self):
        super(res34, self).__init__()
        self.net = models.resnet34(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output

class cnn_svm_dataset(data.Dataset):
    def __init__(self, csv_path='./coords/G_TwoTypes_train.csv',
                 label_dict = {'GBM': 0, 'LGG': 1, },
                 transform=None):
        coords = pd.read_csv(csv_path)[:5]

        slides_path = coords['Path'].to_list()
        slides_label = coords['TypeName'].to_list()

        patch_level_label = []
        grid = []
        wsi_indexs = [0]
        curr_len = 0
        for i, wsi_direc in enumerate(slides_path):
            patch_name_list = os.listdir(wsi_direc)  # g is a slide directory path

            # sampling
            # print(len(patch_name_list), sampleNum

            # add prefix, so the patch has its full path
            temp_full_path = [os.path.join(wsi_direc, x) for x in patch_name_list]

            # prepare WSI index, each number is the begin of a wsi
            curr_len += len(temp_full_path)
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
        self.wsi_label = num_wsi_label
        self.transform = transform
        self.mode = None

    def __getitem__(self, index):
        grid_path, label = self.grid[index], self.patch_labels[index]
        img = Image.open(grid_path)
        target = label
        img = img.resize((224, 224), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.grid)


def extracter(loader, d_set, model, use_gpu = True,):
    """
    extract 1x1000 feature vectors from the patch
    :param loader:
    :param model:
    :param use_gpu:
    :return:
    """
    model.eval()

    data_len = len(d_set)
    print("data length is {}".format(data_len))

    res = torch.FloatTensor(data_len, 512)

    with torch.no_grad():
        for i, (input, label) in enumerate(loader):
            print('Extract\tBatch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = model(input).view((len(input), 512))
            print(output.shape)
            res[i * args.batch_size:i * args.batch_size + input.size(0)] = output.detach()[:].clone()

    return res.cpu().numpy()


def pooling_by3Norm(fea_vector, wsi_indexs):
    assert wsi_indexs[-1] == len(fea_vector)
    print("feature_vec shape", fea_vector.shape)
    res = np.zeros((len(wsi_indexs)-1, fea_vector.shape[1]))
    print(res.shape)

    for i in range(len(wsi_indexs)-1):
        start = wsi_indexs[i]
        end   = wsi_indexs[i+1]
        curr_wsi_vectors = fea_vector[start:end]

        vec_3_norm = np.linalg.norm(x=curr_wsi_vectors, axis=0, keepdims=False)
        # shape: (512,)
        print(vec_3_norm.shape)
        # assert vec_3_norm.shape == (1, fea_vector.shape[1])

        res[i] = vec_3_norm
    return res

def find_topk_features(_train_data, _train_label, fea_num=100):
    labels = list(Counter(_train_label).keys())

    # GBM 1  LGG 0
    set_0 = _train_data[_train_label == 0]
    set_1 = _train_data[_train_label == 1]

    diff = np.sum(set_0, axis=0) - np.sum(set_1, axis=0)
    diff = np.abs(diff)

    # find most distinguish 100 features
    # as the final feature vector
    top_k_index = diff.argsort(diff)[::-1][:fea_num]
    print(top_k_index)
    top_k_index = top_k_index.astype(int)
    print(top_k_index)
    return top_k_index


best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()

    model = res34()
    model.cuda()

    device_ids = range(torch.cuda.device_count())
    print(device_ids)

    # if necessary, mult-gpu training
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model)
    else:
        pass

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    # load data
    train_dset = cnn_svm_dataset(csv_path='./coords/G_TwoTypes_Train.csv', transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_dset = cnn_svm_dataset(csv_path='./coords/G_TwoTypes_Test.csv', transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    train_feaVector = extracter(loader=train_loader, d_set= train_dset, model=model)
    val_feaVector = extracter(loader=val_loader, d_set= val_dset, model=model)
    logger.info("train_feaVector.shape, val_feaVector.shape:" + str(train_feaVector.shape) + ';' +
                str(val_feaVector.shape))
    print(train_feaVector.shape, val_feaVector.shape)

    train_info = {
        'train_array':train_feaVector,
        'train_index':train_dset.wsi_indexs
    }
    val_info = {
        'val_array': val_feaVector,
        'val_index': val_dset.wsi_indexs
    }

    np.save("train_vector.npy", train_info)
    np.save('val_vector.npy', val_info)

    # calculate 3-norm
    train_data = pooling_by3Norm(train_feaVector, train_dset.wsi_indexs)
    test_data  = pooling_by3Norm(val_feaVector, val_dset.wsi_indexs)

    train_label = train_dset.wsi_label
    test_label  = val_dset.wsi_label

    # select most 100 different features as the final feature inputed into SVM
    topk_fea_index = find_topk_features(_train_data=train_data, _train_label=train_label, fea_num=100)

    svm_train_data = train_data[:, topk_fea_index]
    svm_test_data  = test_data[:, topk_fea_index]

    svm_data_all = {
        'train_data':  svm_train_data,
        'train_label': train_label,
        'test_data':   svm_test_data,
        'test_label':  test_label
    }
    np.save('svm_data_all.npy', svm_data_all)

    # train a SVM
    clf = SVC(kernel='rbf')
    print('SVM training')
    clf.fit(svm_train_data, train_label)
    print('SVM predicting')
    pred = clf.predict(svm_test_data)

    acc = (pred == np.array(test_label)).sum() / len(test_label)

    print("WSI level acc using SVM is {}".format(acc))

if __name__ == '__main__':
    main()

