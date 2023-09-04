import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys

import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds==labels.data).item()
    
    average_accuracy = running_corrects/len(test_loader.dataset)
    average_loss = test_loss/len(test_loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Accuracy: {100*average_accuracy}%')
    


def train(model, train_loader, validation_loader, epochs, criterion, optimizer, hook): 
    
    for epoch in range(epochs):
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        hook.set_mode(smd.modes.EVAL)
        model.eval()
        running_corrects = 0
        
        ## validation
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                
        total_accuracy = running_corrects / len(validation_loader.dataset)
        logger.info(f'Validation set: Average accuracy: {100*total_accuracy}%')
        
    return model 
    
def net():
    model = models.resnet50(pretrained = True)
    
    ### fixing the conv layers
    for param in model.parameters():
        param.required_grad = False 
    
    num_features = model.fc.in_features
    num_classes = 5
    
    ### network to train
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256), 
        nn.ReLU(),                 
        nn.Linear(256, 128),
        nn.ReLU(), 
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128,  num_classes),
        nn.LogSoftmax(dim=1)
    )
    return model

def create_data_loaders(datadir, batch_size):

    train_path = os.path.join(datadir, 'train')
    validation_path = os.path.join(datadir, 'validation')
    test_path = os.path.join(datadir, 'test')
    ### some data augmentation and resizing
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.6),
              transforms.Resize(256),
              transforms.Resize((224, 224)),
              transforms.ToTensor()
        ]
    )
    
    ## just resizing
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    
    ## processing data to datasets, just train and validation
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=val_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=val_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, validation_data_loader, test_data_loader

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    model = model.to(device)
    
    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    ## hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    ## process train and validation data
    train_data_loader, val_data_loader, test_data_loader = create_data_loaders(datadir=args.data_dir, batch_size=args.batch_size)
    
    ## train model and validate
    model = train(model, train_data_loader, val_data_loader, args.epochs, loss_criteria, optimizer, hook)
    ## testing
    test(model, test_data_loader, loss_criteria, hook)
    
    ## saving model
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model saved")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="train input batch size")
    parser.add_argument( "--test_batch_size", type=int, default=1000, metavar="N", help="test input batch size")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model")
    
    args=parser.parse_args()
    
    main(args)