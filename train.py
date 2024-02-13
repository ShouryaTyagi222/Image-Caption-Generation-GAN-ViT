import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import logging
from tqdm import tqdm
import numpy as np

from model.model import *
from config import Config
from data.data_loader import *

level = logging.INFO
format_log = '%(message)s'
handlers = [logging.FileHandler('./output/output.log'), logging.StreamHandler()]
logging.basicConfig(level=level, format=format_log, handlers=handlers)

class Trainer:
    def __init__(self,config) -> None:

        self.config= config
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "google/vit-base-patch16-224-in21k"
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

        # Instantiate the modified model
        model_with_intermediate = ViTWithIntermediateOutput.from_pretrained(model_name)

        additional_layers = torch.nn.Sequential(
            torch.nn.Linear(197, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256)  # Adjust num_classes based on your task
        )

        # Instantiate models
        generator = Generator(self.config.max_len,self.config.vocab_size,model_with_intermediate,additional_layers,feature_extractor)
        self.discriminator = Discriminator(self.config.vocab_size)
        self.gan_model = GAN(generator, self.discriminator)

        # Optimizers
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

        # Loss function
        self.g_loss = nn.BCELoss()
        self.d_loss = nn.BCELoss()

        self.train_loader = load_data()

    def train(self):
        logging.info(f' Training '.center(self.terminal_width, '*'))
        # Training loop (you need to replace this with your actual training data)
        num_epochs = self.config.epochs
        for epoch in range(num_epochs):
            logging.info(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            progbar = tqdm(self.train_loader)
            losses_G, losses_D, losses_D_real, losses_D_fake = [], [], [], []

            for i, batch_items in enumerate(progbar):

                image_features = batch_items[0]
                real_captions = batch_items[1]

                pred_real_labels = self.discriminator(real_captions)

                # Forward pass through the GAN
                fake_captions, pred_fake_labels = self.gan_model(image_features)

                # print(fake_captions,pred_fake_labels)
                real_labels = torch.ones_like(pred_fake_labels).float()
                fake_labels = torch.zeros_like(pred_fake_labels).float()

                real_discriminator_loss = self.d_loss(pred_real_labels, real_labels)
                fake_discriminator_loss = self.d_loss(pred_fake_labels, fake_labels)
                total_discriminator_loss = real_discriminator_loss + fake_discriminator_loss

                self.discriminator_optimizer.zero_grad()
                total_discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # Compute generator loss
                if epoch % self.config.train_gen_steps==0:
                    generator_loss = self.g_loss(pred_fake_labels, real_labels)

                    self.generator_optimizer.zero_grad()
                    generator_loss.backward()
                    self.generator_optimizer.step()
                    
                    losses_G.append(generator_loss.cpu().data.numpy())

                losses_D.append(total_discriminator_loss.cpu().data.numpy())
                losses_D_real.append(real_discriminator_loss.cpu().data.numpy())
                losses_D_fake.append(fake_discriminator_loss.cpu().data.numpy())

                progbar.set_description("G = %0.3f, D = %0.3f, R_real = %0.3f, R_fake = %0.3f,  " %
                                        (np.mean(losses_G), np.mean(losses_D),))

            logging.info(f'G = {np.mean(losses_G):.3f}, D = {np.mean(losses_D):.3f}' )

            print(f"Epoch [{epoch}/{num_epochs}] | Generator Loss: {generator_loss.item()} | Discriminator Loss: {total_discriminator_loss.item()}")


if __name__ == "__main__":
    config = Config
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    print('preparing the trainer')
    trainer = Trainer(config)
    print('the training is starting')
    trainer.train()