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
from train import model_arch
from PIL import Image
import json
import os
from prettytable import PrettyTable

#os.environ['QT_QPA_PLATFORM']='offscreen'


def get_input_args():
   
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Input parameters")

    parser.add_argument("--epochs", type = int, default =1 , help = 'Training Epochs')
    parser.add_argument("--hidden_units", type = int, default =1024, help = 'Number Hidden units of the model')
    parser.add_argument("--arch", type = str, default = 'vgg', help = "CNN classifier algorithm to use...Kindly input 'vgg', 'resnet' or 'alexnet'")
    parser.add_argument("--learnrate", type = float, default =0.003, help = 'Learning Rate')
    parser.add_argument("--is_gpu", default =True, help = 'Set device to GPU if true or sets device to CPU if false')
    parser.add_argument("--save_dir", type = str, default ='checkpoint.pth', help = 'Directory where the trained model will be saved')
    parser.add_argument("--predict_image", type = str, default ='flowers/test/11/image_03141.jpg', help = 'The path to the image to be used for prediction')
    parser.add_argument("--topk", type = int, default =5, help = 'Prints the TOP K classes')
    
    return parser.parse_args()


def load_checkpoint(filepath, arch, learnrate):
    model = model_arch(arch)
    checkpoint = torch.load(filepath)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    model.classifier = checkpoint['classifier']
    optimizer.load_state_dict = checkpoint['optimizer_state_dict']
    epochs = checkpoint['epochs']
    
    for params in model.parameters():
        params.requires_grad=False
    
    return model



def load_json():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

#Resize, crops, and normalizes a PIL image for a PyTorch model and returns a Numpy array
def process_image(image):
    image = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    np_image = image_transforms(image).float()
  
    return np_image


#Predict the class (or classes) of an image using a trained deep learning model
def predict(image_path, model, device, topk):
    image = process_image(image_path)
    image = image.to(device)
    model.to(device)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        
        logps = model.forward(image)
        ps = torch.exp(logps)
    
    return ps.topk(topk)



def main():
    in_args = get_input_args()
    
    print("Epochs: ",in_args.epochs)
    print("Hidden Units: ",in_args.hidden_units)
    print("Network Architecture: ",in_args.arch)
    print("Learning Rate: ",in_args.learnrate)
    print("GPU: ",in_args.is_gpu)
    print("SAVE_DIR: ",in_args.save_dir)
    print("Image Directory", in_args.predict_image)
    print("Top K", in_args.topk)
    
    #define the device if GPU is available and user sets TRUE for GPU
    if in_args.is_gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    cat_to_name = load_json()
    
    #load the model to be used for prediction and set the model to evaluation mode
    model = load_checkpoint(in_args.save_dir, in_args.arch, in_args.learnrate)
    model.eval()

    #call the process image function to preprocess the image
    image_path = in_args.predict_image
    image = process_image(image_path)


    #get the probability and classes values by calling the predict function
    probs, classes = predict(image_path, model, device, in_args.topk)
    
    print("Probabilities: \n", probs)
    print("Classes are: \n", classes)

    #convert the output of the predict function to lists
    probs = probs.tolist()[0]
    classes = classes.tolist()[0]


    #convert the probability indices to class labels 
    for key, value in model.class_to_idx.items():
        for i in range(len(classes)):        
            if classes[i] == value:
                classes[i] = key
                classes[i] = cat_to_name[classes[i]]

    
    #initializing table object to print the count of images
    '''my_table = PrettyTable()
    my_table.field_names = ['Probabilities', 'Class Label']
    my_table.add_col([probs, classes])
    print(my_table)'''
    
    for i, j in zip(classes, probs):
        print("{:>15}      {:.3f}".format(i, j))

    
if __name__ == '__main__':
    main()
    
    
    