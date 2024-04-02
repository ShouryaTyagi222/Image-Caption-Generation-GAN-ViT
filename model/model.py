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
        output = self.fc_out(attention_output)

        return output, attention_output

class Generator(nn.Module):
    def __init__(self, max_len, vocab_size, pretrained_model, feature_extractor, device):
        super(Generator, self).__init__()
        self.max_len = max_len
        self.vit = pretrained_model
        self.vit.eval()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_extractor = feature_extractor
        self.additional_layers = torch.nn.Sequential(
                                    torch.nn.Linear(197, 64),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64, 256)
                                )

        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 512, 4, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.device = device

        self.fc_combined = MultiHeadSelfAttention(embed_dim=2*128, num_heads=8, vocab_size=vocab_size).to(device)
        self.diversity_temperature = 1.0

    def forward(self, image_input, targets=None):
        inputs = self.feature_extractor(images=image_input, return_tensors="pt").to(self.device)
        img_features = self.vit(**inputs)
        hidden_states = img_features.hidden_states
        last_layer_features = hidden_states[-1]
        x_pooled = self.global_max_pool(last_layer_features)
        flattened = self.flatten(x_pooled)
        final_img_features = self.additional_layers(flattened)

        sequences = torch.tensor([[1] for _ in range(final_img_features.shape[0])], dtype=torch.long).to(self.device)  # Start token

        # yhat_index, hidden_states = self.get_word(sequences, final_img_features, 0)

        embedded = self.embedding(sequences)
        lstm_out, _ = self.lstm(embedded)
        features = lstm_out[:, -1, :]
        features = self.fc1(features)
        features = self.relu(features)
        features = self.dropout(features)
        final_text_features = self.fc2(features)
        final_text_features = self.relu(final_text_features)

        # print('IMAGE FFEATURES SHAPE : ', final_img_features.shape)
        # print('TEXTUAL FEATURE SHAPES :', final_text_features.shape)

        combined_features = torch.mul(final_text_features, final_img_features)
        yhat, hidden_states = self.fc_combined(combined_features.unsqueeze(0))

        yhat_probs = F.softmax(yhat, dim=2)
        sequences = []
        sequences.append(yhat_probs.squeeze(0))

        for i in range(self.max_len-2):
            yhat, hidden_states = self.fc_combined(hidden_states)
            yhat_probs = F.softmax(yhat, dim=2)
            sequences.append(yhat_probs.squeeze(0))
            
        return torch.cat([t.unsqueeze(1) for t in sequences], dim=1), final_img_features
    
    def get_word(self, sequences, final_img_features, i=-1, target = None):
        embedded = self.embedding(sequences)
        lstm_out, _ = self.lstm(embedded)
        features = lstm_out[:, -1, :]
        features = self.fc1(features)
        features = self.relu(features)
        features = self.dropout(features)
        final_text_features = self.fc2(features)
        final_text_features = self.relu(final_text_features)

        combined_features = torch.cat((final_text_features, final_img_features), dim=1)
        yhat, hidden_states = self.fc_combined(combined_features.unsqueeze(0))

        yhat_probs = F.softmax(yhat, dim=2)

        return yhat_probs, hidden_states

class Discriminator(nn.Module):
    def __init__(self, image_feature_size = 256, embedding_dim = 300, hidden_size = 256, num_layers=1, dropout=0.5):
        super(Discriminator, self).__init__()
        
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.fc_intermediate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_final = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features, sequences, one_hot = True):
        image_output = torch.relu(image_features)

        if one_hot:
            embedded_sequence = self.embedding(sequences.long())
        else:
            b=[]
            for batch in range(sequences.shape[0]):
                s = []
                for seq in range(sequences.shape[1]):
                    # print(seq.shape)
                    s.append(torch.matmul(sequences[batch][seq],self.embedding.weight))
                b.append(s)
                embedded_sequence = torch.stack([torch.stack(inner_list) for inner_list in b], dim=0)
        # print('EMBEDS SHAPE :', embedded_sequence.shape)
        lstm_output, _ = self.lstm(embedded_sequence)
        sequence_output = lstm_output[:, -1, :]  # Use only the last hidden state
        
        combined_features = torch.mul(image_output, sequence_output)
        
        intermediate_output = self.fc_intermediate(combined_features)
        
        output = self.fc_final(intermediate_output)
        output = self.sigmoid(output)
        return output

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
model_with_intermediate.eval()


additional_layers = torch.nn.Sequential(
    torch.nn.Linear(197, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 256)
)

# Freeze parameters of the ViT model
for param in model_with_intermediate.parameters():
    param.requires_grad = False

def load_model(device):
    model_with_intermediate #.to(device)
    gen=Generator(35,8485,model_with_intermediate,feature_extractor, device)
    dis=Discriminator()
    # gen.to(device)
    # dis.to(device)
    # gen.train()
    # dis.train()

    return gen, dis