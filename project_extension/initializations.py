from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from robustness.model_utils import make_and_restore_model
from robustness.datasets import CIFAR
from scipy.fftpack import dct, idct
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, LBFGSAttack
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch.nn as nn
import torch
import itertools
import argparse
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

matplotlib.use('agg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# see whether to use GPU
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
