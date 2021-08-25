import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
import argparse
from workspace_utils import active_session


def get_input_args():
   
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Input parameters")

    parser.add_argument("--epochs", type = int, default =1 , help = 'Training Epochs')
    parser.add_argument("--hidden_units", type = int, default =1024, help = 'Number Hidden units of the model')
    parser.add_argument("--arch", type = str, default = 'vgg', help = "CNN classifier algorithm to use...Kindly input 'vgg', 'resnet' or 'alexnet'")
    parser.add_argument("--learnrate", type = float, default =0.003, help = 'Learning Rate')
    parser.add_argument("--is_gpu", default =True, help = 'Set device to GPU if true or sets device to CPU if false')
    parser.add_argument("--save_dir", type = str, default ='checkpoint_part2.pth', help = 'Directory where the trained model will be saved')
    
    return parser.parse_args()

def data_process():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return trainloader, validloader, testloader, train_datasets

def model_arch(arch):
    if arch.lower() == 'vgg':
        print("\nUsing VGG16 Architecture\n")
        return models.vgg16(pretrained=True)
    elif arch.lower() == 'resnet':
        print("\nUsing RESNET18 Architecture\n")
        return models.resnet18(pretrained=True)
    elif arch.lower() == 'alexnet':
        print("\nUsing ALEXNET Architecture\n")
        return models.alexnet(pretrained=True)
    else:
        return print("Invalid architecture..Kindly input 'vgg', 'resnet' or 'alexnet'")
    
def Network(arch, hidden_units):
    model = model_arch(arch)
    input_size = model.classifier[0].in_features
    
    for params in model.parameters():
        params.requires_grad=False

    #build the network classifier

    input_size = model.classifier[0].in_features
    output_size = 102
    
    my_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size,hidden_units)),
        ('relu',nn.ReLU()),
        ('dropout',nn.Dropout(p=0.3)),
        ('fc2',nn.Linear(hidden_units,output_size)),
        ('output',nn.LogSoftmax(dim=1))]))

    model.classifier = my_classifier
    
    
    return model

def train_model(model, learning_rate, epochs, trainloader, validloader, device):
    #define the loss function
    criterion = nn.NLLLoss()

    #define optimizer to train the model classifier parameters only
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    
    training_loss = 0
    steps = 0
    
    with active_session():
        for e in range(epochs):
            for inputs, labels in trainloader:       
                steps+=1
                #move input images and labels to GPU device
                inputs, labels = inputs.to(device), labels.to(device)

                #set the gradient to zero for every step
                optimizer.zero_grad()

                #perform feed forward of the model with input images
                output = model.forward(inputs)
                #compute loss
                loss = criterion(output, labels)
                #perform backpropagation to calculate gradient for new weights
                loss.backward()
                #update network weights using optimizer steps
                optimizer.step()
                #update the training loss
                training_loss +=loss.item()

                #perform validation at every "print_every" steps
                #if print_every % steps == 0:
            else:
                validate_loss = 0
                accuracy = 0
                #set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    #loop through the validation dataset
                    for inputs, labels in validloader:
                        #move the data to GPU device
                        inputs, labels = inputs.to(device), labels.to(device)
                        #feed forward the data to the model
                        logps = model.forward(inputs)
                        #compute validation loss
                        loss = criterion(logps, labels)
                        #update validation loss
                        validate_loss+=loss.item()

                        #calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print('Epoch: {}/{}...'.format(e+1, epochs),
                     'Steps: {}...'.format(steps),
                     'Training Loss: {:.3f}...'.format(training_loss/len(trainloader)),
                     'Validation Loss: {:3f}...'.format(validate_loss/len(validloader)),
                     'Accuracy: {:.2f}%'.format((accuracy/len(validloader))*100))

                #clear the training loss
                training_loss = 0

                #set model back to traning mode with dropout
                model.train()
    print("End of training loop")       
    #returned trained model after the iterations of the epochs        
    return model, optimizer

def save_checkpoint(model, optimizer, epochs, save_dir, train_datasets):
    checkpoint = {'classifier': model.classifier,
              'epochs': epochs,
              #'optimizer': optimizer
              'optimizer_state_dict': optimizer.state_dict,
              'class_to_idx': train_datasets.class_to_idx,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir)
    
    return checkpoint

#main function where all the methods will be called and executed
def main():
    in_args = get_input_args()

    print("Epochs: ",in_args.epochs)
    print("Hidden Units: ",in_args.hidden_units)
    print("Network Architecture: ",in_args.arch)
    print("Learning Rate: ",in_args.learnrate)
    print("GPU: ",in_args.is_gpu)
    print("SAVE_DIR: ",in_args.save_dir)
    
    
    #load the data from the data process function
    trainloader, validloader, testloader, train_datasets = data_process()
    
    #define the device if GPU is available and user sets TRUE for GPU
    if in_args.is_gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    #Load network model
    model = Network(in_args.arch, in_args.hidden_units)
    
    #Train model
    model, optimizer = train_model(model, in_args.learnrate, in_args.epochs, trainloader, validloader, device)
    
    #save checkpoint
    checkpoint = save_checkpoint(model, optimizer, in_args.epochs, in_args.save_dir, train_datasets)
    
    
if __name__ == '__main__':
    main()
    
    
    