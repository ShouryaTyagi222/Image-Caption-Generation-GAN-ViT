import torch
from torchvision import transforms
from ImageEncoder import ImageEncoder

import numpy as np

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
