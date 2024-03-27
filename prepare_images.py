import torch
from transformers import ViTFeatureExtractor, ViTModel
from torch import nn
import torchvision
from PIL import Image
import os
import numpy as np
import json
import cv2
import pickle
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from datasets import Sequence, Value, Array2D, Array3D
from nltk.translate.bleu_score import corpus_bleu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 

import cv2 
import os

import pickle
with open('./flickr8k.pkl', 'rb') as f:
  d=pickle.load(f)

max_len=d['max_len']
model_data=d['data']

image_folder = './Images'
output_folder = './preprocessed_images'
images = os.listdir(image_folder)
for image_name in tqdm(images):
  image_path= os.path.join(image_folder, image_name)
  img = cv2.imread(image_path)
  print('image shape :',img.shape)
  img = cv2.resize(img,(224,224))
  print('saving pat :', os.path.join(image_folder,image_name))
  cv2.imwrite(os.path.join(image_folder,image_name),img)