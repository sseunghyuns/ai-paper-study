import torch
import wandb
import random
import numpy as np
from args import Args
from train import train_fn
from models.unet import UNet
from dataloader import NuclieData
from models.vgg11 import VGGSegmentation
from torch.utils.data import  DataLoader
from util import get_train_transform, get_valid_transform

def main(args, wandb):
    device = args["DEVICE"]

    # Load data
    train_dataset = NuclieData(path='data/pp_train_imgs', transform=get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=args["BATCH_SIZE"], shuffle=True, num_workers=2, drop_last=True)

    valid_dataset = NuclieData(path='data/pp_valid_imgs', transform=get_valid_transform())
    valid_loader = DataLoader(valid_dataset, batch_size=args["BATCH_SIZE"], shuffle=False, num_workers=2, drop_last=False)  

    # Load model
    model_type = args["MODEL"]
    if model_type == 'my_unet':
        model = UNet().to(device)
    else:
        model = VGGSegmentation().to(device)

    # Train
    train_fn(args, model, train_loader, valid_loader, wandb)


if __name__ == "__main__":

    args = Args().params
    wandb.init(project="unet", entity="sseunghyun", name = f"{args['MODEL']}")
    wandb.config.update({
    "Model": args["MODEL"],
    })

    random_seed = args['RANDOM_SEED']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args, wandb)