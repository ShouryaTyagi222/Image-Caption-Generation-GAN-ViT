
import torch.optim as optim
import torch
from torch.nn.utils import clip_grad_norm_
from torch import nn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F
import os
import pickle as pkl
import numpy as np

from model import load_model
from data_loader import load_data
from config import *
device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")

from rouge_score import rouge_scorer
import numpy as np
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


print('INITIALIZING THE MODEL TRAINING')
gen, dis = load_model(device)
gen= gen.to(device)
dis= dis.to(device)


for i in gen.named_parameters():
    print(i)