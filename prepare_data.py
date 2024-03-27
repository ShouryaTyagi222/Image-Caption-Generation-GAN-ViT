from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import json
import os
from tqdm import tqdm

BASE_DIR='/data/circulars/DATA/layoutLM+Tactful/model_outputs/gcmi/temp'

with open(os.path.join(BASE_DIR,'captions.txt'),'r') as f:
    next(f)
    captions_doc=f.read()

mapping={}
for line in tqdm(captions_doc.split('\n')):
    tokens=line.split(',')
    if len(line)<2:
        continue
    image_id,caption=tokens[0],tokens[1:]
    image_id=image_id.split('.')[0]
    caption=' '.join(caption)
    if image_id not in mapping:
        mapping[image_id]=[]
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption=captions[i]

            caption=caption.lower()

            caption=caption.replace('[^A-Za-z]','')
            caption=caption.replace('\s+',' ')
            caption='startseq '+' '.join([word for word in caption.split() if len(word)>1])+' endseq'
            captions[i]=caption

clean(mapping)

all_captions=[]
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
all_captions.append('PAD')

tokenizer=Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size=len(tokenizer.word_index)+1
print('VOCAB SIZE :',vocab_size)

word_index = tokenizer.word_index
# print(word_index)

print(word_index['pad'])
pad_index = word_index['pad']

max_length=max(len(caption.split()) for caption in all_captions)
print('MAX LENGTH :',max_length)

data=[]
for key in mapping.keys():
  captions=mapping[key]
  print('Image Name :',key)
  for caption in captions:
      #encoder the sequece
      seq=tokenizer.texts_to_sequences([caption])[0]
      seq.extend([pad_index]*(max_length-len(seq)))
      print('Tokenized Sequence :',seq)
      data.append({
        'image_name':key,
        'caption':seq,
      })

import pickle
with open('flickr8k_2.pkl', 'wb') as f:
    pickle.dump({'data':data,'tokenizer':tokenizer,'vocab_size':vocab_size,'max_len':max_length}, f)