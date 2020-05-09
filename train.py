import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# ------------------------------------------------------------------------------- #
# Define Functions
# ------------------------------------------------------------------------------- #
# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str,
                        default='vgg13',
                        choices=['densenet121', 'alexnet','vgg16'],
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        default="./",
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001,
                        help='Define gradient descent learning rate as float')

    parser.add_argument('--epochs', 
                        type=int, 
                        default=8,
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true",
                        default=True,
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    
    if(args.arch=='help'):
        print('Available networks:')
        print('- densenet121')
        print('- vgg13')
        print('- vgg16')
        print('- alexnet')
        quit()
        
    if(not(args.learning_rate>0 and args.learning_rate<1)):
        print('Invalid learning rate:')
        print('must be a value between 0 and 1')
        quit()
        
    if(args.epochs<0):
        print('Invalid epoch number ')
        print('must be greater than 0')
        quit()
    archi=['densenet121', 'vgg16', 'vgg13', 'alexnet']
    
    if args.arch not in archi:
        print('Invalid architecture received')
        print('Type \"python train.py --arch help"\ for more details')
        quit()
        
    return args

# Function train_transformer(train_dir) performs training transformations on a dataset
def train_transformer(train_dir):
    
   # Define transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data
# Function test_transformer(test_dir) performs test/validation transformations on a dataset
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
# Function data_loader(data, train=True) creates a dataloader from dataset imported
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader

# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# primaryloader_model(architecture="vgg16") downloads model (primary) from torchvision
def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    exis_models={
                'densenet121': models.densenet121(pretrained=True),
                'vgg13': models.vgg13(pretrained=True),
                'vgg16': models.vgg16(pretrained=True),
                'alexnet': models.alexnet(pretrained=True)}
    
    model = exis_models.get(architecture)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

# Function initial_classifier(model, hidden_units) creates a classifier with the corect number of input layers
def initial_classifier(model, archi):
    # Check that hidden layers has been input
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    #print(input_features)
    if archi=='densenet121':
    # Define Classifier
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, 1024, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.4)),
                              ('fc2', nn.Linear(1024, 102, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    if archi=='vgg13' or archi=='vgg16':
    # Define Classifier
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, 4096, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.4)),
                              ('fc2', nn.Linear(4096, 102, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    elif archi=='alexnet':
    # Define Classifier
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, 2048, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.4)),
                              ('fc2', nn.Linear(2048, 102, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return classifier

# Function validation(model, testloader, criterion, device) validates training against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# Function network_trainer represents the training of the network model
def network_trainer(Model, Trainloader, validloader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps):

    print("Training process initializing .....\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            Optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(Model, validloader, Criterion, Device)
            
                print("Epoch: {}/{} | ".format(e+1, Epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/Print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
            
                running_loss = 0
                Model.train()

    return Model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, archi, Train_data, epoch):
       
    # Save model at checkpoint
    
    if isdir(Save_Dir):
        # Create `class_to_idx` attribute in model
        Model.class_to_idx = Train_data.class_to_idx

        # Create checkpoint dictionary
        checkpoint = {'architecture': archi,
                      'classifier': Model.classifier,
                      'class_to_idx': Model.class_to_idx,
                      'state_dict': Model.state_dict(),
                      'epochs': epoch
                     }
        if Save_Dir!=None:
            file=Save_Dir+'/'+'latest_checkpoint.pth'
        else:
            file='latest_checkpoint.pth'
        # Save checkpoint
        torch.save(checkpoint, file)
    else:
        print('Directory doesn\'t exist')
       
# =============================================================================
# Main Function
# =============================================================================

# Function main() is where all the above functions are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = initial_classifier(model,
                                          archi=args.arch)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 50
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Quickly Validate the model
    validate_model(trained_model, testloader, device)
    
    # Save the model
    initial_checkpoint(trained_model, args.save_dir, args.arch, train_data,  args.epochs)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()