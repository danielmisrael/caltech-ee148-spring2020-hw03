from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.embed(x)
        x = F.log_softmax(x, dim=1)
        return x

    def embed(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x



def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def compute_accuracy(model, device, loader):
    '''
    Helper method for printing accuracy scores.
    '''
    model.eval()    # Set the model to inference mode
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, test_num,
            100. * correct / test_num))

def compute_loss(model, device, loader):
    '''
    Helper method for computing loss.
    '''
    model.eval()    # Set the model to inference mode
    test_loss = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_num += len(data)
    test_loss /= test_num
    return test_loss

def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    return test_loss

def plot_features(model, classes, data_loader):
    labels = []
    images = []
    representation = []
    model.eval()
    with torch.no_grad():
        for (x,y) in data_loader:
            out = model.embed(x).data.numpy()
            representation.append(out)
            y = y.data.numpy()
            labels.extend(y)
            images.extend(x)

    representation = np.concatenate(representation, axis = 0)

    tsne = TSNE(n_components = 2, perplexity = 50)
    vis_rep = tsne.fit_transform(representation)

    plt.figure(figsize=(8, 8))
    for i in range(classes):
        plt.scatter(vis_rep[np.array(labels)==i,1],
                    vis_rep[np.array(labels)==i,0], s = 3)
    plt.title("TSNE Visualization of MNIST Representations")
    plt.legend([str(i) for i in range(classes)])
    plt.show()
    return vis_rep, images

def euclidean_dist(i, j, ip, jp):
    return np.sqrt((i - ip) ** 2 + (j-jp) ** 2)


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    # Flag will augment training data
    parser.add_argument('--augment', action='store_true', default=False,
                        help='For applying data augmentation')

    # Flag runs the vary training size experiment
    parser.add_argument('--vary-training', action='store_true', default=False,
                        help='For varying training size')

    parser.add_argument('--analyze-model', action='store_true', default=False,
                        help='Analyze what network has learned')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        # model = fcNet().to(device)
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    if args.augment:
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([       # Data preprocessing
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),           # Add data augmentation here
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    else:

        # Pytorch has default MNIST dataloader which loads data at each iteration
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([       # Data preprocessing
                        transforms.ToTensor(),           # Add data augmentation here
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))



    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    split = int(0.8 * len(train_dataset))
    subset_indices_train = range(split)
    subset_indices_valid = range(split, len(train_dataset))

    if args.vary_training:
        splits = [split]
        for subset in [1/2, 1/4, 1/8, 1/16]:
            splits.append(int(split * subset))
        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        training_loss = []
        test_loss = []
        for s in splits:
            subset_indices_train = range(s)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                sampler=SubsetRandomSampler(subset_indices_train)
            )

            model = Net().to(device)

            # Try different optimzers here [Adam, SGD, RMSprop]
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

            # Set your learning rate scheduler
            scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

            # Training loop

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                scheduler.step()    # learning rate scheduler

            training_loss.append(compute_loss(model, device, train_loader))
            test_loss.append(test(model, device, test_loader))

        plt.plot(splits, training_loss, label='Training Loss')
        plt.plot(splits, test_loss, label='Test Loss')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Training Images')
        plt.ylabel('Loss')
        plt.title('Loss Against Training Set Size')
        plt.show()
        return

    if args.analyze_model:
        assert os.path.exists(args.load_model)

        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        ## find 9 pictures of misclassified images
        images = []
        classified =[]
        predictions = []
        targets = []
        with torch.no_grad():   # For the inference step, gradient is not computed
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)

                i = 0
                for p, t in zip(pred.squeeze(), target):
                    predictions.append(p)
                    targets.append(t)
                    if p != t and len(images) < 9:
                        images.append(data[i])
                        classified.append(p)
                    i += 1
        figure = plt.figure(figsize=(8, 16))
        cols, rows = 3, 3
        for i in range(rows):
            for j in range(cols):
                figure.add_subplot(rows, cols, i*cols+j+1)
                plt.axis("off")
                plt.imshow(images[i*cols+j].squeeze())
                plt.title("Classified: " + str(classified[i*cols+j].numpy()))
        plt.show()

        #visualize kernels
        kernels1 = model.conv1.weight.detach().clone()
        figure = plt.figure(figsize=(8, 16))
        cols, rows = 3, 3
        for i in range(rows):
            for j in range(cols):
                figure.add_subplot(rows, cols, i*cols+j+1)
                plt.axis("off")
                plt.imshow(kernels1[i*cols+j].squeeze())
        plt.suptitle('Kernel Visualizations')
        plt.show()

        # Confusion Matrix
        mat = confusion_matrix(targets, predictions)
        plt.imshow(mat)
        plt.title('Confusion Matrix')
        plt.xlabel('True Labels')
        plt.ylabel('Predictions')
        plt.colorbar()
        plt.show()

        # Visualize representation
        closest = []
        vis_rep, test = plot_features(model, 10, test_loader)

        for i in range(5):
            closest.append(test[i])
            v = np.array([euclidean_dist(vis_rep[i][0], vis_rep[i][1], coord[0], coord[1]) for coord in vis_rep])
            idx = np.argsort(v)

            for index in idx[1:9]:
                closest.append(test[index])

        figure = plt.figure(figsize=(8, 16))
        cols, rows = 9, 5
        for i in range(rows):
            for j in range(cols):
                figure.add_subplot(rows, cols, i*cols+j+1)
                plt.axis("off")
                plt.imshow(closest[i*cols+j].squeeze())
        plt.suptitle("Embedding Nearest Neighbors")
        plt.show()
        return




    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    # model = ConvNet().to(device)
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    training_loss = []
    val_loss = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        training_loss.append(compute_loss(model, device, train_loader))
        val_loss.append(compute_loss(model, device, val_loader))
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    print("Final accuracy scores")
    print("Training:")
    compute_accuracy(model, device, train_loader)
    print("Validation:")
    compute_accuracy(model, device, val_loader)

    epochs = list(range(1, args.epochs + 1))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
