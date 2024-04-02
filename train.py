
import torch.optim as optim
import torch
from torch.nn.utils import clip_grad_norm_
from torch import nn
from tqdm import tqdm
import warnings
from torchviz import make_dot
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

torch.autograd.set_detect_anomaly(True)

def set_gradients(model, grad = True):
    for param in model.parameters():
        print('Before :',param.grad)
        print('Requires Grad :',param.requires_grad)
        param.requires_grad = grad
        print('After :', param.grad)

def cal_rouge_scores(r_caption, f_caption):
    rouge1_p=[]
    rouge1_r=[]
    rouge1_f=[]
    rouge2_p=[]
    rouge2_r=[]
    rouge2_f=[]
    rougel_p=[]
    rougel_r=[]
    rougel_f=[]

    for hypothesis, reference in zip(r_caption, f_caption):
        scores = scorer.score(hypothesis, reference)
        rouge1_p.append(scores['rouge1'].precision)
        rouge1_r.append(scores['rouge1'].recall)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_p.append(scores['rouge2'].precision)
        rouge2_r.append(scores['rouge2'].recall)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougel_p.append(scores['rougeL'].precision)
        rougel_r.append(scores['rougeL'].recall)
        rougel_f.append(scores['rougeL'].fmeasure)
    
    rouge1_p=np.mean(rouge1_p)
    rouge1_r=np.mean(rouge1_r)
    rouge1_f=np.mean(rouge1_f)
    rouge2_p=np.mean(rouge2_p)
    rouge2_r=np.mean(rouge2_r)
    rouge2_f=np.mean(rouge2_f)
    rougel_p=np.mean(rougel_p)
    rougel_r=np.mean(rougel_r)
    rougel_f=np.mean(rougel_f)

    return rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f

with open(os.path.join(OUTPUT_DIR, 'model_logs.txt'), 'a') as f:
    f.write('INITIALIZING THE MODEL AND DATALOADER\n')

if os.path.exists(os.path.join(OUTPUT_DIR,'model.pkl')):
    print('CONTINUING THE MODEL')
    with open(os.path.join(OUTPUT_DIR,'model.pkl'), 'rb') as f:
        model = pkl.load(f)

    gen=model['gen'].to(device)
    dis=model['dis'].to(device)
    start_epoch=model['epoch']

else:
    print('INITIALIZING THE MODEL TRAINING')
    gen, dis = load_model(device)
    # gen= gen.to(device)
    # dis= dis.to(device)
    start_epoch = 0
dataloader, tokenizer = load_data(DATA_FILE,IMAGE_DIR, BATCH_SIZE)
gen= gen.to(device)
# print(gen)
dis= dis.to(device)
word_index = tokenizer.index_word
# print(word_index)

def decode(seq):
    decoded_seq = []
    for token in seq :
        # print('TOKEN : ',token.item())
        if token.item() != 0:
            decoded_seq.append(word_index[token.item()])
        else:
            decoded_seq.append('PAD')
    decoded_seq = ' '.join(decoded_seq)
    # print('Decoded_seq :',decoded_seq)
    return decoded_seq

class WassersteinLoss(nn.Module):
    def __init__(self, end_token=2):
        super(WassersteinLoss, self).__init__()
        self.end_token = end_token

    def forward(self, real_distribution, generated_distribution):
        batch_size = real_distribution.size(0)
        seq_length = real_distribution.size(1)

        wasserstein_distance = 0

        for i in range(batch_size):
            for j in range(seq_length):
                if real_distribution[i, j] == self.end_token:
                    # Stop calculation when end token is encountered
                    break

                wasserstein_distance += torch.abs(real_distribution[i, j] - generated_distribution[i, j])

        return wasserstein_distance
# criterion = WassersteinLoss()
    
def batch_decode(model_output):
    final_batch = []
    for batch in model_output:
        final_seq = []
        for text in batch:
            # print(text.shape)
            _, yhat_index = torch.max(text, dim=0)
            final_seq.append(yhat_index)
        final_batch.append(final_seq)
    return final_batch

# g_loss = WassersteinLoss()
g_loss = nn.BCELoss()
d_loss = nn.BCELoss()

generator_optimizer = optim.Adam(gen.parameters(), lr=G_LEARNING_RATE)
discriminator_optimizer = optim.Adam(dis.parameters(), lr=D_LEARNING_RATE)

print(device)

smoother = SmoothingFunction()

def bleu_score(predictions, targets):
    # Convert predictions and targets to lists of strings

    return corpus_bleu(targets.split(), predictions.split(), smoothing_function=smoother.method1)

with open(os.path.join(OUTPUT_DIR, 'model_logs.txt'), 'a') as f:
    f.write('STARTING THE TRAINIING\n')

for epoch in range(EPOCH):
    progbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch')
    losses_G, losses_D, losses_D_real, losses_D_fake, bleu_scores = [], [], [], [], []
    gen.train()
    dis.train()
    r_captions = []
    f_captions = []


    for i, batch_items in enumerate(progbar):
        image_features = batch_items[0].to(device)
        real_captions = batch_items[1].to(device)
        real_captions = real_captions[:, 1:]

        fake_captions, final_img_features = gen(image_features, real_captions)
        fake_captions_tokenized = torch.argmax(fake_captions, dim=2)

        # graph = make_dot(fake_captions, params=dict(gen.named_parameters()))
        # graph.render("computation_graph_NEW")  # Save as PDF

        # print(fake_captions)
        r_caption = decode(real_captions[0])
        f_caption = decode(fake_captions_tokenized[0])
        # print(f_caption)
        # print(r_caption)

        pred_fake_labels = dis(final_img_features, fake_captions, one_hot=False)

        real_labels = torch.ones_like(pred_fake_labels).float()
        fake_labels = torch.zeros_like(pred_fake_labels).float()

        generator_loss = g_loss(pred_fake_labels, real_labels)
        b_score = bleu_score(r_caption,f_caption)
        total_generator_loss=generator_loss
        # set_gradients(gen)
        generator_optimizer.zero_grad()
        generator_loss.backward()
        # clip_grad_norm_(gen.parameters(), max_norm=0.5)  # Adjust max_norm as needed
        generator_optimizer.step()
        # count = 0
        # for param in gen.parameters():
        #     # print(param.grad)
        #     if param.grad is not None:
        #         count+=1
        #         # print(f"Gradient AAA for {param}: {param.grad.norm()}")

        # print(count)

        # break
        pred_real_labels = dis(final_img_features.detach(), real_captions)
        pred_fake_labels = dis(final_img_features.detach(), fake_captions.detach(), one_hot = False)
        real_discriminator_loss = d_loss(pred_real_labels, real_labels)
        fake_discriminator_loss = d_loss(pred_fake_labels, fake_labels)
        total_discriminator_loss = real_discriminator_loss + fake_discriminator_loss
        discriminator_optimizer.zero_grad()
        total_discriminator_loss.backward()
        discriminator_optimizer.step()

        losses_G.append(total_generator_loss.cpu().data.numpy())
        losses_D.append(total_discriminator_loss.cpu().data.numpy())
        losses_D_real.append(real_discriminator_loss.cpu().data.numpy())
        losses_D_fake.append(fake_discriminator_loss.cpu().data.numpy())
        bleu_scores.append(b_score)
        # r_captions.append(r_caption)
        # f_captions.append(f_caption)


        progbar.set_description("EPOCH = %s/%s, G_LOSS = %0.3f, D_LOSS = %0.3f, current_loss = %0.3f, " %(epoch, EPOCH, np.mean(losses_G), np.mean(losses_D), total_generator_loss.item()))

    # break 
    # rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f = cal_rouge_scores(r_captions, f_captions)
    # print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f},\n')

    with open(os.path.join(OUTPUT_DIR, 'model_logs.txt'), 'a') as f:
        f.write("EPOCH = %s/%s, G_LOSS = %0.3f, D_LOSS = %0.3f, D_REAL = %s, D_FAKE = %s, BLEU_SCORE = %s\n REAL_CAPTION = %s \n FAKE_CAPTION = %s\n" %(epoch, EPOCH, np.mean(losses_G), np.mean(losses_D),np.mean(losses_D_real), np.mean(losses_D_fake), np.mean(bleu_scores), r_caption, f_caption))
    # with open(os.path.join(OUTPUT_DIR, 'model_logs.txt'), 'a') as f:
    #     f.write(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f},\n')
        


    print('REAL CAPTION :',r_caption)
    print('FAKE CAPTION :',f_caption)

    with open(os.path.join(OUTPUT_DIR,'model.pkl'), 'wb') as f:
        pkl.dump({'gen': gen,'dis': dis,'epoch':start_epoch+epoch+1}, f, protocol=pkl.HIGHEST_PROTOCOL)