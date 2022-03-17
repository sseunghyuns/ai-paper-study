import torch
import wandb
import random
import numpy as np
from tqdm import tqdm
from args import Args
import torch.nn as nn
from model import ResNet34
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
            ),
        ])

def train(args, wandb):

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size = args["BATCH_SIZE"], shuffle=True, num_workers=2)

    valid_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=get_transforms())
    valid_loader = DataLoader(valid_dataset, batch_size = args["BATCH_SIZE"], shuffle=False, drop_last=True, num_workers=2)

    if args["MODEL"] == 'resnet34':
        model = ResNet34(in_channels=3, out_channels=10)
        model.cuda()
    else:
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["LEARNING_RATE"])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    best_loss = 1e9
    device = args["DEVICE"]

    for epoch in range(1, args["NUM_EPOCHS"]+1):

        total_loss = []
        num_correct, num_samples = 0, 0
        model.train()
        for i, (X, y) in enumerate(tqdm(train_loader, total=len(train_loader))):
            
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            preds = torch.argmax(outputs, dim=-1)

            num_correct += sum(preds==y).sum()
            num_samples += preds.shape[0]

            loss = criterion(outputs, y)
            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy =  num_correct/num_samples
        train_loss = sum(total_loss)/len(total_loss)

        with torch.no_grad():
            model.eval()

            val_losses = []
            num_correct, num_samples = 0, 0

            for X, y in tqdm(valid_loader, total=len(valid_loader)):
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                preds = torch.argmax(outputs, dim=-1)
                num_correct += (preds==y).sum()
                num_samples += preds.shape[0]

                val_losses.append(loss.item())

            val_accuracy = num_correct/num_samples
            val_loss = sum(val_losses)/len(val_losses)
            torch.cuda.empty_cache()

            print("Epoch: {}/{}.. ".format(epoch, 50) +
                        "Training Accuracy: {:.4f}.. ".format(train_accuracy) +
                        "Training Loss: {:.4f}.. ".format(train_loss) +
                        "Valid Accuracy: {:.4f}.. ".format(val_accuracy) + 
                        "Valid Loss: {:.4f}.. ".format(val_loss))

            wandb.log({'Train accuracy': train_accuracy, 'Train loss': train_loss, 'Valid accuracy': val_accuracy, 'Valid Loss': val_loss})

            if val_loss < best_loss:
                print("Valid loss improved from {:.4f} -> {:.4f}".format(best_loss, val_loss))
                best_loss = val_loss
                earlystopping_counter = 0

            else:
                earlystopping_counter += 1
                print("Valid loss did not improved from {:.4f}.. Counter {}/{}".format(best_loss, earlystopping_counter, 7))
                if earlystopping_counter > 7:
                    print("Early Stopped ...")
                    break
        
        scheduler.step(val_loss)


if __name__ == '__main__':

    args = Args().params

    wandb.init(project="classification-models", entity="sseunghyun", name = f"{args['MODEL']}")
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

    train(args, wandb)