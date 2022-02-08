import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from PIL import Image
from dataset import dataset
from generator import *
from discriminator import *
from new_trainer import *
from train_options import *

def generate_image(epoch_id):
    transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                        ])

    test_image_path = Config.validate_image_path
    test_image = transform(Image.open(test_image_path).convert("L")).view(1, 1, 128, 128).to("cuda")
    res_image = gen_mask(test_image)
    res_image = res_image.detach().cpu() 
    save_image(res_image, f"./{epoch_id}.jpg")

Config = TrainOptions().gather_options()

train_dataset = dataset(root_Mask = Config.mask_root, root_X_ray = Config.x_ray_root)
train_dataloader = DataLoader(train_dataset, batch_size = Config.batch_size, 
        sampler = RandomSampler(train_dataset), num_workers = Config.num_workers, pin_memory = True)

torch.cuda.empty_cache()

gen_mask, gen_x_ray = Generator(Config.in_channels, Config.num_residual_blocks).to(Config.device), Generator(Config.in_channels, Config.num_residual_blocks).to(Config.device)
dis_mask, dis_x_ray = Discriminator(Config.in_channels).to(Config.device), Discriminator(Config.in_channels).to(Config.device)

optimizer_g = optim.Adam(params = list(gen_mask.parameters()) + list(gen_x_ray.parameters()),
                        lr = Config.learning_rate, betas = (Config.beta1, 0.999), )
optimizer_d = optim.Adam(params = list(dis_mask.parameters()) + list(dis_x_ray.parameters()),
                        lr = Config.learning_rate, betas = (Config.beta1, 0.999), )

l1, mse = nn.L1Loss(), nn.MSELoss()

trainer = Trainer(optimizer_g, l1, optimizer_d, mse)

for epoch in range(1, Config.num_epochs + Config.num_epochs_decay + 1):
    print(f"Training Epoch {epoch}")
    trainer.train_one_epoch(train_dataloader, (gen_mask, gen_x_ray), (dis_mask, dis_x_ray))
    
    if Config.validate_image_path is not None and epoch % Config.validate_frequency == 0:
        print("Generating image ...")
        generate_image(epoch)

    print("\n")

trainer.save_checkpoint((dis_mask, dis_x_ray), (gen_mask, gen_x_ray))