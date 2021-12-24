# Imports for the project
import os
import time
import torch
import json
import numpy as np
import argparse, sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models


# Get user input data
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir')
parser.add_argument('--arch')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_units')
parser.add_argument('--epochs')
parser.add_argument('--gpu')
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                   }

# Load the datasets with ImageFolder
directories = {'train': train_dir, 
               'valid': valid_dir, 
               'test' : test_dir}

image_datasets = {data: datasets.ImageFolder(directories[data], transform=data_transforms[data])
                  for data in ['train', 'valid', 'test']}

# Set default values if user did not enter any value
if (arch == "vgg13"):
    input_size = 25088
    output_size = 102
elif (arch == "densenet121"):
    input_size = 1024
    output_size = 102
else:
    print("Please select model architectures vgg13 or densenet121 (Eg.python train.py data_dir --arch \"vgg13\")")
    exit()

if save_dir is None:
    save_dir = "checkpoint.pth"
    
if learning_rate is None:
    learning_rate = 0.001
else:
    learning_rate = float(learning_rate)

if hidden_units is None:
    if (arch == "vgg13"):
        hidden_units = 4096
    elif (arch == "densenet121"):
        hidden_units = 500
else:
    hidden_units = int(hidden_units)
    
if epochs is None:
    epochs = 5
else:
    epochs = int(epochs)

if device is None:
    device = "cpu"
    
if(data_dir == None) or (save_dir == None) or (arch == None) or (learning_rate == None) or (hidden_units == None) or (epochs == None) or (device == None):
    print("data_dir or arch or learning_rate or hidden_units, and epochs cannot be none")
    exit()


# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {data: torch.utils.data.DataLoader(image_datasets[data], batch_size=32, shuffle=True) for data in ['train', 'valid', 'test']} 

# Build and train network
if (arch == 'vgg13'):
    model = models.vgg13(pretrained=True)
elif (arch == 'densenet121'):
    model = models.densenet121(pretrained=True)
model

# Validation on the test set
for param in model.parameters():
    param.requires_grad = False

# Build a feedforward network
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                        ('fc2', nn.Linear(hidden_units, output_size)),
                                        ('output', nn.LogSoftmax(dim=1))]))

# Put the classifier on the pretrained network
model.classifier = classifier

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device)
print("Training model started")
for epoch in range(epochs):

    for dataset in ['train', 'valid']:
        if dataset == 'train':
            model.train()  
        else:
            model.eval()   
        
        running_loss = 0.0
        running_accuracy = 0
        
        for inputs, labels in dataloaders[dataset]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(dataset == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward 
                if dataset == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)
        
        dataset_sizes = {data: len(image_datasets[data]) for data in ['train', 'valid', 'test']}
        epoch_loss = running_loss / dataset_sizes[dataset]
        epoch_accuracy = running_accuracy.double() / dataset_sizes[dataset]
        
        print("\nEpoch: {}/{}... ".format(epoch+1, epochs),
              "{} Loss: {:.4f}    Accurancy: {:.4f}".format(dataset, epoch_loss, epoch_accuracy))
        
# Validation on the test set
def check_accuracy_on_test(test_loader):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images: %d %%' % (100 * correct / total))
check_accuracy_on_test(dataloaders['train'])

# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'model': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)
print("Model saved to:" + save_dir)