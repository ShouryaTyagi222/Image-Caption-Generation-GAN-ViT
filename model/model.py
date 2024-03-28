import torch
from transformers import ViTFeatureExtractor, ViTModel
from torch import nn
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import random

from ..config import *

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
        # print(attention_output.shape)
        output = self.fc_out(attention_output)  # Adjusted output

        return output, attention_output

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

        yhat_index, hidden_states = self.get_word(sequences, final_img_features, 0)
        sequences = torch.cat((sequences, yhat_index.T), dim=1)

        # # Teacher forcing
        # if targets is not None and random.random() < TEACHER_FORCING_RATIO:
        #     for i in range(self.max_len - 1):
        #         yhat, hidden_states = self.fc_combined(hidden_states)
        #         yhat_probs = F.softmax(yhat, dim=2)
        #         _, yhat_index = torch.max(yhat_probs, dim=2)
        #         sequences = torch.cat((sequences, yhat_index.T), dim=1)
        # else:
        for i in range(self.max_len - 2):
            yhat, hidden_states = self.fc_combined(hidden_states)
            yhat_probs = F.softmax(yhat, dim=2)
            _, yhat_index = torch.max(yhat_probs, dim=2)
            sequences = torch.cat((sequences, yhat_index.T), dim=1)
            # print(sequences)

        return sequences
    
    def get_word(self, sequences, final_img_features, i, target = None):
        embedded = self.embedding(sequences)
        lstm_out, _ = self.lstm(embedded)
        features = lstm_out[:, 0, :]
        features = self.fc1(features)
        features = self.relu(features)  # Apply ReLU activation
        features = self.dropout(features)  # Apply dropout
        final_text_features = self.fc2(features)
        final_text_features = self.relu(final_text_features)  # Apply ReLU activation

        combined_features = torch.cat((final_text_features, final_img_features), dim=1)
        yhat, hidden_states = self.fc_combined(combined_features.unsqueeze(0))

        yhat_probs = F.softmax(yhat, dim=2)
        _, yhat_index = torch.max(yhat_probs, dim=2)

        return yhat_index, hidden_states

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