import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import pickle

class Flickr8kDataset(Dataset):
    def __init__(self, file, image_dir):
        with open(file, 'rb') as f:
            pkl_file = pickle.load(f)
        self.data_dict=pkl_file['data']
        self.max_len=pkl_file['max_len']
        self.image_dir=image_dir

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # print('getting the data :',idx)
        img_name = self.data_dict[idx]['image_name']+'.jpg'
        caption = np.array(self.data_dict[idx]['caption'])
        # caption = np.hstack([caption,np.zeros((self.max_len - len(caption),))])

        image = cv2.imread(os.path.join(self.image_dir,img_name))
        # image = cv2.resize(image,(224,224))

        caption = torch.tensor(caption).int()
        image = torch.tensor(image).float()
        image = image.permute(2, 0, 1)

        return image, caption
    

def load_data(data_file,img_dir,batch_size):

    # Create an instance of the Flickr8kDataset
    flickr_dataset = Flickr8kDataset(data_file,img_dir)
    with open(data_file, 'rb') as f:
        pkl_file = pickle.load(f)
    tokenizer=pkl_file['tokenizer']

    num_workers = 1  # Adjust based on your system's capabilities
    dataloader = DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader, tokenizer