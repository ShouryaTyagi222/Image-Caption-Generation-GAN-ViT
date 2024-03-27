import torch
from transformers import ViTFeatureExtractor, ViTModel
from torch import nn
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import random

from config import *

# config = Config()


model_name = "google/vit-base-patch16-224-in21k"

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, vocab_size):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Adjusted input and output size
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.fc_out(attention_output)  # Adjusted output

        return output

class Generator(nn.Module):
    def __init__(self, max_len, vocab_size, pretrained_model, additional_layers, feature_extractor, device):
        super(Generator, self).__init__()
        self.max_len = max_len
        self.vit = pretrained_model
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_extractor = feature_extractor
        self.additional_layers = additional_layers

        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 512, 4, batch_first=True, dropout=0.2)  # Increase LSTM hidden size and add dropout
        self.fc1 = nn.Linear(512, 256)  # Add additional fully connected layer
        self.fc2 = nn.Linear(256, 128)  # Add another fully connected layer
        self.relu = nn.ReLU()  # Add ReLU activation function
        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        self.device = device

        # Move modules to the specified device
        self.embedding.to(device)
        self.lstm.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.relu.to(device)
        self.dropout.to(device)

        self.fc_combined = MultiHeadSelfAttention(embed_dim=2*128, num_heads=8, vocab_size=vocab_size).to(device)
        # Increase the number of attention heads to 8 for increased complexity
        # Define diversity temperature
        self.diversity_temperature = 1.0  # Adjust this value to control the diversity level

    def forward(self, image_input, targets=None):
        inputs = self.feature_extractor(images=image_input, return_tensors="pt").to(self.device)
        img_features = self.vit(**inputs)
        hidden_states = img_features.hidden_states
        last_layer_features = hidden_states[-1]
        x_pooled = self.global_max_pool(last_layer_features)
        flattened = self.flatten(x_pooled)
        final_img_features = self.additional_layers(flattened)

        # sequences = torch.tensor([[1] + [0 for _ in range(self.max_len - 1)] for _ in range(final_img_features.shape[0])], dtype=torch.long).to(self.device)
        sequences = torch.tensor([[1] for _ in range(final_img_features.shape[0])], dtype=torch.long).to(self.device)  # Start token

        # sequences = self.next_word(sequences, final_img_features, i)


        # Generate the sequence for each position
        # Teacher forcing
        for i in range(self.max_len - 1):
            if targets is not None and random.random() < TEACHER_FORCING_RATIO:
                sequences = self.next_word(sequences, final_img_features, i, targets[:, i+1])
            else:
                sequences = self.next_word(sequences, final_img_features, i)

        return sequences
    
    def next_word(self, sequences, final_img_features, i, target = None):
        embedded = self.embedding(sequences)
        lstm_out, _ = self.lstm(embedded)
        # print(lstm_out.shape,lstm_out[:, i, :].shape)
        features = lstm_out[:, i, :]
        features = self.fc1(features)
        features = self.relu(features)  # Apply ReLU activation
        features = self.dropout(features)  # Apply dropout
        final_text_features = self.fc2(features)
        final_text_features = self.relu(final_text_features)  # Apply ReLU activation

        combined_features = torch.cat((final_text_features, final_img_features), dim=1)
        yhat = self.fc_combined(combined_features.unsqueeze(0))

        yhat_probs = F.softmax(yhat, dim=2)
        _, yhat_index = torch.max(yhat_probs, dim=2)

        # Update the sequence for all images at once
        # sequences[:, i + 1] = yhat_index.squeeze().detach()
        if target is not None:
            sequences = torch.cat((sequences, target.unsqueeze(1)), dim=1)
        else:
            sequences = torch.cat((sequences, yhat_index.T), dim=1)
        # print(sequences)
        return sequences

        # sequences = []

        # with torch.no_grad():
        #     for image in final_img_features:
        #         image = image.unsqueeze(0).to(self.device)
        #         # sequence = torch.tensor([1] + [0 for _ in range(self.max_len - 1)], dtype=torch.long).unsqueeze(0).to(self.device)
        #         sequence = torch.tensor([1], dtype=torch.long).unsqueeze(0).to(self.device)
        #         # print('image_features :',image.shape)
        #         for i in range(self.max_len - 1):
        #             embedded = self.embedding(sequence)
        #             lstm_out, _ = self.lstm(embedded)
        #             features = lstm_out[:, -1, :]
        #             final_text_features = self.fc(features)

        #             # print(i,'-> caption_features :',final_text_features.shape)
        #             combined_features = torch.cat((final_text_features, image), dim=1)
        #             # print(combined_features)

        #             yhat = self.fc_combined(combined_features.unsqueeze(0))
        #             # print('yhat :',yhat)

        #             # yhat_with_noise = yhat[0] + torch.randn_like(yhat[0]) / self.diversity_temperature
        #             # yhat_softmax = F.softmax(yhat_with_noise, dim=1)
        #             # # Sample from the softmax distribution to introduce randomness
        #             # yhat_index = torch.multinomial(yhat_softmax, 1).squeeze(1)
                    
        #             yhat_probs = F.softmax(yhat, dim=2)
        #             # Then you might select the word with the highest probability:
        #             _, yhat_index = torch.max(yhat_probs, dim=2)

        #             # Update the sequence with the sampled index
        #             # sequence[0][i + 1] = yhat_index.item()
        #             sequence = torch.cat((sequence, yhat_index), dim=1)

        #             # Check for end token and break the loop if found
        #             if yhat_index.item() == 2:  # Assuming end token index is 2
        #                 break

        #         sequences.append(sequence[0])

        # return torch.stack(sequences)

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

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)

class ViTWithIntermediateOutput(ViTModel):
    def forward(self, pixel_values, return_dict=True, output_hidden_states=True, **kwargs):
        return super().forward(pixel_values, return_dict=return_dict, output_hidden_states=output_hidden_states, **kwargs)

model_with_intermediate = ViTWithIntermediateOutput.from_pretrained(model_name)
model_with_intermediate.train()


additional_layers = torch.nn.Sequential(
    torch.nn.Linear(197, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 128)
)


def load_model(device):
    model_with_intermediate.to(device)
    gen=Generator(35,8485,model_with_intermediate,additional_layers,feature_extractor, device)
    dis=Discriminator(8485)
    gen.to(device)
    dis.to(device)
    gen.train()
    dis.train()

    return gen, dis