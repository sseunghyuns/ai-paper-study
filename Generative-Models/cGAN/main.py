import os
import torch
import wandb
import shutil
import torchvision
from args import Args
import torch.nn as nn
from utils import sample_image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Generator, Discriminator


def get_transforms():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.5],
                std  = [0.5]
            )
        ])


def train(args, wandb):

    if os.path.exists("images"):
        shutil.rmtree("images", )
    os.makedirs("images")


    device = args["DEVICE"]
    num_classes = args["NUM_CLASSES"]
    latent_dim = args["LATENT_DIM"]

    # Dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', download=True, train=True, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=args["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=2)

    # Generator and Discriminator 
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Adversarial loss
    criterion = nn.BCELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args["LEARNING_RATE"], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args["LEARNING_RATE"], betas=(0.5, 0.999))

    # Training
    for epoch in range(1, args["NUM_EPOCHS"]):
        for idx, (imgs, labels) in enumerate(train_loader):
            
            batch_size = imgs.shape[0]

            # Discriminator inputs - real images and labels(0~9)
            real_imgs = imgs.view(batch_size, -1).to(device)       # images -> Flatten image   [batch_size, 784]
            labels = F.one_hot(labels, num_classes=num_classes).type(torch.FloatTensor).to(device)  # labels -> one-hot vectors [batch_size, num_classes]
            
            # Discriminator inputs - labels(real(1) or fake(0))
            valid_y = torch.ones(batch_size, 1).to(device)   # [batch_size, 1]
            fake_y = torch.zeros(batch_size, 1).to(device)   # [batch_size, 1]

            ##########################
            ### Training Generator ###
            ##########################
            generator.train()
            optimizer_g.zero_grad()

            # Generator inputs - Noise(z) and labels 
            z = torch.randn(batch_size, latent_dim).to(device)  # [batch_size, 100]
            gen_labels = torch.randint(num_classes, (batch_size,)).to(device) # [batch_size]
            gen_labels = F.one_hot(gen_labels, num_classes=num_classes).type(torch.FloatTensor).to(device) # [batch_size, num_classes]

            gen_imgs = generator(z, gen_labels) # generate fake images
            validity = discriminator(gen_imgs, gen_labels) # Does these gen_imgs fool discriminator?
            loss_g = criterion(validity, valid_y) # Generator should minimize this loss = Discriminator should predict gen_imgs as 1.
            
            loss_g.backward()
            optimizer_g.step() # Update Generator
            
            ##############################
            ### Training Discriminator ###
            ##############################
            optimizer_d.zero_grad()

            # Loss for real_imgs
            preds_real = discriminator(real_imgs, labels)
            real_loss_d = criterion(preds_real, valid_y)

            preds_fake = discriminator(gen_imgs.detach(), gen_labels)
            fake_loss_d = criterion(preds_fake, fake_y)

            loss_d = (real_loss_d + fake_loss_d)/2
            loss_d.backward()
            optimizer_d.step()

            if idx % 100 ==0:
                print("Epoch: {}/{}.. ".format(epoch, args["NUM_EPOCHS"]) +
                    "Step: [{}/{}].. ".format(idx, len(train_loader)) +
                    "Generator Loss: {:.4f}.. ".format(loss_g.item()) +
                    "Discriminator Loss: {:.4f}.. ".format(loss_d.item()))
                
                wandb.log({'Generator Loss': loss_g.item(), 'Discriminator Loss': loss_d.item()})

        sample_image(n_row=10, epoch=epoch, generator=generator)



if __name__ == '__main__':

    args = Args().params

    wandb.init(project="generative-models", entity="sseunghyun", name = f"{args['MODEL']}")
    wandb.config.update({
    "Model": args["MODEL"],
    })

    train(args, wandb)