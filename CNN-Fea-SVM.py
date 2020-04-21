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

# Use ResNet-34 to extract features
#

model = models.resnet34(True)
print(model.fc)