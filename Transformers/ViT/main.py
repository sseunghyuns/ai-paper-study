import os
import torch
import torch.nn as nn
from tqdm import tqdm

import argparse
from input_patch import PatchDataset
from eval_functions import evaluation

def parse_args():
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--save_acc', default=50, type=int, help='val acc') # acc 50 이상만 저장
    parser.add_argument('--epochs', default=501, type=int, help='training epoch')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='drop rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--latent_dim', default=128, type=int, help='latent dimension') # D
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads') # attention head
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer')
    parser.add_argument('--dataname', default='cifar10', type=str, help='data name')
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='GPU usage')

    args = parser.parse_args()
    
    return args

def run(args):
    latent_dim = args.latent_dim
    mlp_hidden_dim = int(latent_dim/2) # Encoder 내 MLP의 hidden layer 노드 수 
    num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size)) # HW/P^2
    device = args.device
    
    if not os.path.exists('./saved'):
        os.makedirs('./saved')
    
    # Load Dataset
    dataset = PatchDataset(patch_size = args.patch_size,
                           img_size = args.img_size,
                           batch_size = args.batch_size)
    
    train_loader, valid_loader, test_loader = dataset.load_dataset()
    
    # Load ViT 
    model = None
    
    best_accuracy = args.save_acc
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        running_loss, num_correct, num_samples = 0, 0, 0
        
        model.train()
        for i, (X, y) in enumerate(tqdm(train_loader, total=len(train_loader))):
            X = X.to(device)
            y = y.to(device)
            
            outputs, _ = model(X)
            loss = criterion(outputs, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            preds = torch.argmax(outputs, dim=-1)
            num_correct += sum(preds==y).sum()
            num_samples += preds.shape[0]
            running_loss += loss.item()

        train_accuracy =  num_correct/num_samples
        train_loss = running_loss/len(train_loader)
        
        # Evaluation
        with torch.no_grad():
            model.eval()
            valid_accuracy, valid_loss = evaluation(model, valid_loader, criterion, device)
            

    print("Epoch: {}/{}.. ".format(epoch, 50) +
                "Training Accuracy: {:.4f}.. ".format(train_accuracy) +
                "Training Loss: {:.4f}.. ".format(train_loss) +
                "Valid Accuracy: {:.4f}.. ".format(valid_accuracy) + 
                "Valid Loss: {:.4f}.. ".format(valid_loss))

    if valid_accuracy > best_accuracy:
        print("Valid Acc improved from {:.4f} -> {:.4f}".format(best_accuracy, valid_accuracy))
        best_accuracy = valid_accuracy
        
        # 기존 경로 제거
        try:
            os.remove(save_path)
        except:
            pass
        save_path = "saved/epoch{}_.tar".format(epoch+1)
    
        torch.save(model.state_dict(), save_path)

    
if __name__ == '__main__':
    args = parse_args()
    # run(args)
