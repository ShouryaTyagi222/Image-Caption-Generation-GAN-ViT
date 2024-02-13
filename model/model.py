import torch
from transformers import ViTFeatureExtractor, ViTModel
from torch import nn
import warnings
warnings.filterwarnings("ignore")


class Generator(nn.Module):
    def __init__(self, max_len, vocab_size, pretrained_model, additional_layers, feature_extractor):
        super(Generator, self).__init__()
        self.max_len = max_len
        self.vit=pretrained_model
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_extractor=feature_extractor
        self.additional_layers = additional_layers

        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 64, 8, batch_first=True)
        self.fc = nn.Linear(64, 256)
        self.fc_combined = nn.Linear(2 * 256, vocab_size)

    def forward(self, image_input):
        inputs = self.feature_extractor(images=image_input, return_tensors="pt")
        img_features = self.vit(**inputs)
        hidden_states = img_features.hidden_states
        last_layer_features=hidden_states[-1]
        x_pooled = self.global_max_pool(last_layer_features)
        flattened=self.flatten(x_pooled)
        final_img_features = self.additional_layers(flattened)

        sequence = torch.tensor([1] + [0 for _ in range(self.max_len - 1)], dtype=torch.long).unsqueeze(0)
        for i in range(self.max_len - 1):
            embedded = self.embedding(sequence)
            lstm_out, _ = self.lstm(embedded)
            features = lstm_out[:, -1, :]  # Use the last hidden state as the feature representation
            final_text_features = self.fc(features)

            # Concatenate image and text features
            combined_features = torch.cat((final_img_features, final_text_features), dim=1)

            # Pass through the combined linear layer for final output
            yhat = self.fc_combined(combined_features)

            yhat_index = torch.argmax(yhat, dim=1)
            sequence[0][i + 1] = yhat_index.item()

            if yhat_index.item() == 2:
                break

        return sequence

class Discriminator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        output = self.fc(hidden[-1, :, :])
        return torch.sigmoid(output)

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        generated_images = self.generator(z)
        discriminator_output = self.discriminator(generated_images)
        return generated_images, discriminator_output
    

class ViTWithIntermediateOutput(ViTModel):
            def forward(self, pixel_values, return_dict=True, output_hidden_states=True, **kwargs):
                return super().forward(pixel_values, return_dict=return_dict, output_hidden_states=output_hidden_states, **kwargs)