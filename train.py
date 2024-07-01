import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from datasets.MPII import MPIIDataset
from vit_models.model import ViTPose
from configs.ViTPose_common import save_checkpoint, load_checkpoint, adjust_learning_rate

# Training 및 Validation 함수
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets, target_weight, meta) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        target_weight = target_weight.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch [{epoch}], Iter [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)

def validate_model(model, val_loader, criterion):
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets, target_weight, meta) in enumerate(val_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            target_weight = target_weight.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()

    return validation_loss / len(val_loader)

def main():
    root_path = '/Users/jeonseung-u/Desktop/DeepLearning/ViTPose/datasets/mpii'
    dataset = MPIIDataset(root_path=root_path, data_version='train')

    # Train-validation split
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ViTPose()
    model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, epoch)
        val_loss = validate_model(model, val_loader, criterion)

        print(f"Epoch [{epoch}] Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False)

if __name__ == '__main__':
    main()
