# -*- coding: utf-8 -*-
from collections import OrderedDict
from torch import nn
import torch
from torch.autograd import Variable
from transformers import ViTFeatureExtractor, ViTModel
from torchvision import transforms
import numpy as np

import json
import numpy as np
import os
import string

from PIL import Image as PIL_Image
from pycocotools.coco import COCO
from utils import explorer_helper as exh

try:
    from urllib.request import urlretrieve, urlopen
except ImportError:
    from urllib import urlretrieve
    from urllib2 import urlopen
import urllib

from socket import error as SocketError
import errno

MAX_SIZE = 0

MAIN_FOLDER = "cocodataset-1"
IMAGE_FOLDER = "{}/images".format(MAIN_FOLDER)
CAPTIONS_FOLDER = "{}/captions".format(MAIN_FOLDER)
LINKS_FOLDER = "{}/links".format(MAIN_FOLDER)
FEATURES_FOLDER = "{}/features".format(MAIN_FOLDER)

IMAGE_FILE = IMAGE_FOLDER + "/COCO_{}2014_{}.jpg"
ID_STR_IMAGE = "COCO_{}2014_{}.jpg"
CAPTIONS_FILE = CAPTIONS_FOLDER + "/{}"
LINKS_FILE = LINKS_FOLDER + "/{}"
FEATURES_FILE = FEATURES_FOLDER + "/{}"

COCO_LINK = "http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
ZIP_NAME = "captions_train-val2014.zip"
TRAIN_FILE = "/data/circulars/DATA/layoutLM+Tactful/model_outputs/gcmi/temp2/gan-image-captioning/cocodataset-1/annotations/captions_train2014.json"
VAL_FILE = "/data/circulars/DATA/layoutLM+Tactful/model_outputs/gcmi/temp2/gan-image-captioning/cocodataset-1/annotations/captions_val2014.json"

FEATS = dict()

def prepare_directories():
    exh.create_directory(MAIN_FOLDER)
    exh.create_directory(IMAGE_FOLDER)
    exh.create_directory(CAPTIONS_FOLDER)
    exh.create_directory(LINKS_FOLDER)
    exh.create_directory(FEATURES_FOLDER)


model_name = "google/vit-base-patch16-224-in21k"
class ImageEncoder(object):
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)

        class ViTWithIntermediateOutput(ViTModel):
            def forward(self, pixel_values, return_dict=True, output_hidden_states=True, **kwargs):
                return super().forward(pixel_values, return_dict=return_dict, output_hidden_states=output_hidden_states, **kwargs)

        self.model_with_intermediate = ViTWithIntermediateOutput.from_pretrained(model_name)
        self.model_with_intermediate
        for param in self.model_with_intermediate.parameters():
            param.requires_grad = False

        self.flatten = nn.Flatten()

    def get(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        img_features = self.model_with_intermediate(**inputs)
        hidden_states = img_features.hidden_states
        last_layer_features = hidden_states[-1]
        fl = self.flatten(last_layer_features)

        return fl[:, :2048]


class ImageFactory(object):
    def __init__(self, resize=None, crop=None):

        self.feature_extractor = ImageEncoder()

        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(_transforms)

    def get_features(self, img):
        img = self.transform(img).cuda()
        return self.feature_extractor.get(img.unsqueeze(0)).cpu().data.numpy().squeeze().astype(np.float32)



def format_caption(caption):
    norm_caption = "".join([w for w in caption if w not in string.punctuation])
    norm_caption = norm_caption.replace("  ", " ").lower()
    if norm_caption[-1] == " ":
        norm_caption = norm_caption[:-1]

    return norm_caption.split('\n')[0]


def extract_features(image_id, set_name, image_factory):
    if not image_id in FEATS:
        img = PIL_Image.open(IMAGE_FILE.format(set_name, '0'*(12-len(str(image_id)))+str(image_id)))
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        feats = np.array(image_factory.get_features(img)).squeeze()
        FEATS[image_id] = feats
    return FEATS[image_id]

def format_set(set_name, filename, image_factory):
    set_file = exh.load_json(filename)
    captions = []
    links = []
    features = []
    beam = dict()

    coco = COCO(filename)

    tot = len(set_file['annotations'])
    el = 1

    for row in set_file['annotations']:
        print("{}/{}".format(el, tot))
        el += 1
        image_id = row['image_id']
        str_id = ID_STR_IMAGE.format(set_name, '0' * (12-len(str(image_id))) + str(image_id))
        is_ok = download_image(image_id, set_name, coco)

        if is_ok:
            caption = format_caption(row['caption'])
            captions.append(caption)
            print(caption)
            links.append(str_id)

            feats = extract_features(image_id, set_name, image_factory)
            features.append(feats)

            if set_name == "val":
                if str_id in beam:
                    beam[str_id]["captions"].append(caption)
                else:
                    beam[str_id] = {
                        "captions": [caption],
                        "feats": feats
                    }
        else:
            print('not ok')

    if MAX_SIZE > 0:
            captions = captions[:MAX_SIZE]
    captions = '\n'.join(captions)
    exh.write_text(captions, CAPTIONS_FILE.format("{}.en".format(set_name)))
    if MAX_SIZE > 0:
            links = links[:MAX_SIZE]
    links = '\n'.join(links)
    exh.write_text(links, LINKS_FILE.format("{}.txt".format(set_name)))
    if MAX_SIZE > 0:
            features = features[:MAX_SIZE]
    features = np.array(features)
    np.save(FEATURES_FILE.format(set_name), features)

    if set_name == "val":
        captions = []
        links = []
        features = []

        for k, v in beam.items():
            links.append(str(k))
            captions.append(v['captions'])
            features.append(v['feats'])

        captions = ["###".join(sentences) for sentences in captions]
        captions = '\n'.join(captions)
        if MAX_SIZE > 0:
            captions = captions[:MAX_SIZE]
        exh.write_text(captions, CAPTIONS_FILE.format("beam.en"))
        if MAX_SIZE > 0:
            links = links[:MAX_SIZE]
        links = '\n'.join(links)
        exh.write_text(links, LINKS_FILE.format("beam.txt"))
        if MAX_SIZE > 0:
            features = features[:MAX_SIZE]
        features = np.array(features)
        np.save(FEATURES_FILE.format("beam"), features)

def run():
    prepare_directories()

    image_factory = ImageFactory(resize=256,crop=224)

    print("Formatting train set")
    format_set("train", TRAIN_FILE, image_factory)
    print("Formatting val set")
    format_set("val", VAL_FILE, image_factory)    

if __name__ == "__main__":
    run()
