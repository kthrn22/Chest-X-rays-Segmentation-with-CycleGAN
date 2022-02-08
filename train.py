import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from train_options import *

Config = TrainOptions().gather_options()

def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 1 - Config.num_epochs) / float(Config.num_epochs_decay + 1)
    return lr_l

class Trainer():
    def __init__(self, optimizer_g, criterion_g, optimizer_d, criterion_d):
        self.optimizer_g = optimizer_g
        self.criterion_g = criterion_g
        self.scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda = lambda_rule)

        self.optimizer_d = optimizer_d
        self.criterion_d = criterion_d
        self.scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda = lambda_rule)

    def loss_discriminator(self, generators, discriminators, images, fake_images):
        gen_mask, gen_x_ray = generators
        dis_mask, dis_x_ray = discriminators
        dis_mask.train()
        dis_x_ray.train()
        
        masks, x_rays = images
        fake_masks, fake_x_rays = fake_images

        dis_mask_real, dis_mask_fake = dis_mask(masks), dis_mask(fake_masks)
        real_loss, fake_loss = self.criterion_d(dis_mask_real, torch.ones_like(dis_mask_real)), self.criterion_d(dis_mask_fake, torch.zeros_like(dis_mask_fake))
        dis_mask_loss = (real_loss + fake_loss) / 2
        dis_mask_loss.backward()

        dis_x_ray_real, dis_x_ray_fake = dis_x_ray(x_rays), dis_x_ray(fake_x_rays)
        real_loss, fake_loss = self.criterion_d(dis_x_ray_real, torch.ones_like(dis_x_ray_real)), self.criterion_d(dis_x_ray_fake, torch.zeros_like(dis_x_ray_fake))
        dis_x_ray_loss = (real_loss + fake_loss) / 2
        dis_x_ray_loss.backward()

        del fake_masks, fake_x_rays, dis_mask_real, dis_x_ray_real, dis_mask_fake, dis_x_ray_fake

        return dis_mask_loss, dis_x_ray_loss

    def loss_generator(self, generators, discriminators, images, fake_images):
        gen_mask, gen_x_ray = generators
        dis_mask, dis_x_ray = discriminators
        gen_mask.train()
        gen_x_ray.train()

        masks, x_rays = images
        fake_masks, fake_x_rays = fake_images
        
        dis_mask_fake, dis_x_ray_fake = dis_mask(fake_masks), dis_x_ray(fake_x_rays)
        adversarial_loss_mask, adversarial_loss_x_ray = self.criterion_d(dis_mask_fake, torch.ones_like(dis_mask_fake)), self.criterion_d(dis_x_ray_fake, torch.ones_like(dis_x_ray_fake))

        cycle_masks, cycle_x_rays = gen_mask(fake_x_rays), gen_x_ray(fake_masks)
        cycle_loss_mask, cycle_loss_x_ray = self.criterion_g(cycle_masks, masks), self.criterion_g(cycle_x_rays, x_rays)

        identity_masks, identity_x_rays = gen_mask(masks), gen_x_ray(x_rays)
        identity_loss_mask, identity_loss_x_ray = self.criterion_g(identity_masks, masks), self.criterion_g(identity_x_rays, x_rays)

        loss = adversarial_loss_mask + adversarial_loss_x_ray + Config.lambda_cycle * (cycle_loss_mask + cycle_loss_x_ray) + Config.lambda_identity * (identity_loss_mask + identity_loss_x_ray)
        loss.backward()

        del fake_masks, fake_x_rays, dis_mask_fake, dis_x_ray_fake, cycle_masks, cycle_x_rays, identity_masks, identity_x_rays

        return loss

    def update_learning_rate(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def train_one_epoch(self, dataloader, generators, discriminators):
        pbar = tqdm(dataloader, total = len(dataloader))
        generator_total_loss, discriminator_total_mask_loss, discriminator_total_x_ray_loss = 0, 0, 0
        gen_mask, gen_x_ray = generators
        dis_mask, dis_x_ray = discriminators

        for batch_idx, batch in enumerate(pbar):
            masks = batch['mask'].to(Config.device)
            x_rays = batch['x_ray'].to(Config.device)

            for param in dis_mask.parameters():
                param.requires_grad = False
            for param in dis_x_ray.parameters():
                param.requires_grad = False

            self.optimizer_g.zero_grad()
            generator_loss = self.loss_generator(generators, discriminators, (masks, x_rays),
                    (gen_mask(x_rays), gen_x_ray(masks)))
            generator_total_loss += generator_loss.item()
            if (batch_idx + 1) % Config.g_accumulation_steps == 0:
                self.optimizer_g.step()

            for param in dis_mask.parameters():
                param.requires_grad = True
            for param in dis_x_ray.parameters():
                param.requires_grad = True

            self.optimizer_d.zero_grad()
            discriminator_mask_loss, discriminator_x_ray_loss = self.loss_discriminator(generators, 
        discriminators, (masks, x_rays), (gen_mask(x_rays), gen_x_ray(masks)))
            discriminator_total_mask_loss += discriminator_mask_loss.item()
            discriminator_total_x_ray_loss += discriminator_x_ray_loss.item()
            if (batch_idx + 1) % Config.d_accumulation_steps == 0:
                self.optimizer_d.step()

            del masks, x_rays

            pbar.set_postfix({"Discriminator Mask loss ": discriminator_total_mask_loss / (batch_idx + 1),
        "Discriminator X-ray loss ": discriminator_total_x_ray_loss / (batch_idx + 1), 
        "Generator loss ": generator_total_loss / (batch_idx + 1)})
    
        self.update_learning_rate()

    def save_checkpoint(self, discriminators, generators):
        dis_mask, dis_x_ray = discriminators
        gen_mask, gen_x_ray = generators

        torch.save({
            "discriminator_mask": dis_mask.state_dict(),
            "discriminator_x_ray": dis_x_ray.state_dict(),
            "discriminator_optimizer": self.optimizer_d.state_dict(),
            "generator_mask": gen_mask.state_dict(),
            "generator_x_ray": gen_x_ray.state_dict(),
            "generator_optimizer": self.optimizer_g.state_dict(),
            "generator_scheduler": self.scheduler_g.state_dict(),
            "discriminator_scheduler": self.scheduler_d.state_dict(),
            "learning_rate": self.optimizer_g.param_groups[0]['lr'],
        }, "checkpoint.pt")