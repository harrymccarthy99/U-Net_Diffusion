import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from datasets import load_dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
from google.colab import drive
import matplotlib.pyplot as plt
import cv2
from google.colab import drive
import time as time_calc
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import pandas as pd
import os
