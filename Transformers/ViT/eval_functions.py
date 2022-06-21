import torch
import torch.nn as nn
from tqdm import tqdm

def evaluation(model, valid_loader, criterion, device):
    running_loss, num_correct, num_samples = 0, 0, 0
    
    for X, y in tqdm(valid_loader, total=len(valid_loader)):
        X = X.to(device)
        y = y.to(device)
        
        outputs, _ = model(X)
        loss = criterion(outputs, y)
        
        preds = torch.argmax(outputs, dim=-1)
        num_correct += sum(preds==y).sum()
        num_samples += preds.shape[0]
        running_loss += loss.item()

    val_accuracy =  num_correct/num_samples
    val_loss = running_loss/len(valid_loader)
    
    return val_accuracy, val_loss
