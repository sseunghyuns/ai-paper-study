import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import warnings
from util import *
warnings.filterwarnings("ignore")

def train_fn(args, model, train_loader, valid_loader, wandb):
    device = args["DEVICE"]
        
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["LEARNING_RATE"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args["SCHEDULDER_PATIENCE"], verbose=True)

    earlystopping_counter = 0
    earlystopping_patience = args["EARLYSTOPPING_PATIENCE"]
    
    best_loss = 1e9

    for epoch in range(1, args["NUM_EPOCHS"]+1):
        total_loss = []
        num_correct, num_pixels = 0, 0

        model.train()
        for i, (imgs, masks) in enumerate(tqdm(train_loader, total=len(train_loader))):
            
            imgs = imgs.to(device)
            masks = masks.to(device)

            pred_masks = model(imgs)
            preds = ((pred_masks) > 0.5).float()
            
            # acc
            num_correct += (preds==masks).sum()
            num_pixels += torch.numel(preds)

            loss = criterion(pred_masks, masks) # [B, 1, 128, 128]

            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = num_correct/num_pixels
        train_loss = sum(total_loss)/len(total_loss)

        with torch.no_grad():
            model.eval()

            valid_loss = []
            num_correct, num_pixels = 0, 0

            for imgs, masks in tqdm(valid_loader, total=len(valid_loader)):
                imgs = imgs.to(device)
                masks = masks.to(device) 

                pred_masks = model(imgs)
                preds = ((pred_masks) > 0.5).float()
                
                # acc
                num_correct += (preds==masks).sum()
                num_pixels += torch.numel(preds)

                loss = criterion(pred_masks, masks)

                valid_loss.append(loss.item())

            valid_acc = num_correct/num_pixels
            valid_loss = sum(valid_loss)/len(valid_loss)

        print("Epoch: {}/{}.. ".format(epoch, args["NUM_EPOCHS"]) +
            "Training Loss: {:.4f}.. ".format(train_loss) +
            "Training Acc: {:.4f}.. ".format(train_acc) +
            "Valid Loss: {:.4f}.. ".format(valid_loss) +
            "Valid Acc: {:.4f}.. ".format(valid_acc))   
        
        wandb.log({'Train accuracy': train_acc, 'Train Loss': train_loss, 'Valid accuracy': valid_acc, 'Valid Loss': valid_loss})

        # Save Model
        if valid_loss < best_loss:
            print("Valid loss improved from {:.4f} -> {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
                }      
            # 기존 경로 제거
            try:
                os.remove(save_path)
            except:
                pass
            save_path = "result/{}_Epoch{}_{:.4f}.tar".format(args["MODEL"], epoch, best_loss)
            
            torch.save(checkpoint, save_path)
            earlystopping_counter = 0

        else:
            earlystopping_counter += 1
            print("Valid loss did not improved from {:.4f}.. Counter {}/{}".format(best_loss, earlystopping_counter, earlystopping_patience))
            if earlystopping_counter > earlystopping_patience:
                print("Early Stopped ...")
                break
        
        scheduler.step(valid_loss)