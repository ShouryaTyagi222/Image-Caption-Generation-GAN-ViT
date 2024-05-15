# -*- coding: utf-8 -*-
from collections import OrderedDict
from torch import nn
import torch
from torch.autograd import Variable
from transformers import ViTFeatureExtractor, ViTModel

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