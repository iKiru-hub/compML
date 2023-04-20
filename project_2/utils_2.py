# standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# pytorch
from torch import nn, optim, no_grad
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from IPython.display import clear_output


""" DATASET """

def get_mnist_data_tf(plot=False):
    
    """
    Function to get the MNIST dataset

    Parameters
    ----------
    plot : bool
        if True, plot the first 10 images of the training set. Default is False

    Returns
    -------
    train_data : torch.Tensor
        training data
    train_labels : torch.Tensor
        training labels
    test_data : torch.Tensor
        test data
    test_labels : torch.Tensor
        test labels
    """

    # Get the data
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # Convert the data to float32
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    # Convert the data to tensors
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)

    # Add a channel dimension
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # Plot the first 10 images of the training set
    if plot:
        plt.figure(figsize=(10, 10))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(train_data[i][0], cmap='gray')
            plt.axis('off')
            plt.title(train_labels[i].item())
        plt.show()

    return train_data, train_labels, test_data, test_labels


def get_mnist_data_torch(batch_size=128, plot=False):

    """
    Function to get the MNIST dataset

    Parameters
    ----------
    batch_size : int
        batch size for the training and test set
    plot : bool
        if True, plot the first 10 images of the training set. Default is False

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        training set
    test_loader : torch.utils.data.DataLoader
        test set
    device : torch.device
        device to use for the training and test set
    """


    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(
        MNIST('./data', train=True, download=True, transform=ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        MNIST('./data', train=False, transform=ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Defining the device
    if torch.cuda.is_available():
        print("<cuda> available, let's exploit the GPU")
        device = torch.device("cuda:0")
    else:
        print("no <cuda> available, let's stick to the CPU")
        device = torch.device("cpu")

    if plot:
        # Plot the first 10 images of the training set
        for i, (images, labels) in enumerate(train_loader):
            if i == 1:
                break
            plt.figure(figsize=(10, 10))
            for j in range(10):
                plt.subplot(1, 10, j+1)
                plt.imshow(images[j][0], cmap='gray')
                plt.axis('off')
                plt.title(labels[j].item())
            plt.show()

    return train_loader, test_loader, device


def add_noise_1(data_x, noise_level=0.9):
    
    """
    Function to add noise to the data

    Parameters
    ----------
    data_x : torch.Tensor
        data to add noise to 
    noise_level : float
        level of noise to add to the data. Default is 0.9

    Returns
    -------
    torch.Tensor
        noisy data
    """

    return torch.clip(data_x + np.random.binomial(1, noise_level, size=data_x.shape) * np.clip(np.random.normal(0.2, 0.2, data_x.size()), 0.1, 0.75), 0, 1)

def one_hot_encoding_in_batches(labels, n_classes=10, batch_size=1024):
    """
    Perform one-hot encoding in smaller batches to avoid memory issues.

    Parameters
    ----------
    labels : torch.Tensor
        Labels to be one-hot encoded.
    n_classes : int
        Number of classes.
    batch_size : int, optional
        Batch size for one-hot encoding. Default is 1024.

    Returns
    -------
    torch.Tensor
        One-hot encoded labels.
    """
    one_hot_labels = []

    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i: i + batch_size]
        one_hot_batch = nn.functional.one_hot(batch_labels.to(torch.int64), n_classes)
        one_hot_labels.append(one_hot_batch)

    return torch.cat(one_hot_labels, dim=0).unsqueeze(dim=1)

def normalize_and_noise(data: np.ndarray, noise_spread: float=0.1, noise_level: float=0.9) -> np.ndarray:

    """
    Function to normalize and add noise to the data

    Parameters
    ----------
    data : np.ndarray
        data to normalize and add noise to
    noise_spread : float
        spread of the noise to add to the data. Default is 0.1
    noise_level : float
        level of noise to add to the data. Default is 0.9

    Returns
    -------
    np.ndarray
        normalized and noisy data
    """

    # normalize the data
    data = data / 255

    # add noise to the data from a binomial distribution and a normal distribution
    return data + np.random.binomial(1, noise_spread, size=data.shape) * np.abs(np.random.normal(0., noise_level, data.shape)).clip(0, 0.5)


# function for getting the data of a given dataset
def build_data(noise_spread=0.1, noise_level=0.9, test_prop=0.1, val_prop=0.1):
    
    """
    Function to get the data from a dataset

    Parameters
    ----------
    noise_spread : float
        spread of the noise to add to the data. Default is 0.1
    noise_level : float
        level of noise to add to the data. Default is 0.9
    test_prop : float
        proportion of the test set with respect to
        the data set. Default is 0.1.
    val_prop : float
        proportion of the validation set with respect to
        the training set. Default is 0.1.

    Returns
    -------
    train_data : torch.Tensor
        training data
    train_labels : torch.Tensor
        training labels
    test_data : torch.Tensor
        test data
    test_labels : torch.Tensor
        test labels
    val_data : torch.Tensor
        validation data
    val_labels : torch.Tensor
        validation labels
    xtrain_noisy : torch.Tensor
        noisy training data
    xtest_noisy : torch.Tensor
        noisy test data
    xval_noisy : torch.Tensor
        noisy validation data
    """

    ### Get the data ###
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    data, labels = mnist['data'], mnist['target']

    # Split the data into train, test and validation sets using train_test_split
    test_size = int(test_prop * len(data))
    val_size = int(val_prop * len(data))
   
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_size)

    # convert the data to numpy arrays
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
    val_data = val_data.to_numpy()
    train_labels = train_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    val_labels = val_labels.to_numpy()

    # Add noise to the training data
    xtrain_noisy = normalize_and_noise(train_data, noise_spread, noise_level)
    xtest_noisy = normalize_and_noise(test_data, noise_spread, noise_level)
    xval_noisy = normalize_and_noise(val_data, noise_spread, noise_level)

    # convert the data to tensors
    train_data = torch.from_numpy(train_data.astype('float32'))
    test_data = torch.from_numpy(test_data.astype('float32'))
    val_data = torch.from_numpy(val_data.astype('float32'))
    train_labels = torch.from_numpy(train_labels.astype('int64'))
    test_labels = torch.from_numpy(test_labels.astype('int64'))
    val_labels = torch.from_numpy(val_labels.astype('int64'))
    xtrain_noisy = torch.from_numpy(xtrain_noisy.astype('float32'))
    xtest_noisy = torch.from_numpy(xtest_noisy.astype('float32'))
    xval_noisy = torch.from_numpy(xval_noisy.astype('float32'))

    # Print the sizes of each data set
    print(f'xtrain: {train_data.size()}')
    print(f'ytrain: {train_labels.size()}')

    print(f'\nxtest: {test_data.size()}')
    print(f'ytest: {test_labels.size()}')

    print(f'\nxval: {val_data.size()}')
    print(f'yval: {val_labels.size()}')
    
    print(f'\nxtrain noisy: {xtrain_noisy.size()}')
    print(f'xtest_noisy: {xtest_noisy.size()}')
    print(f'xval_noisy: {xval_noisy.size()}')

    return train_data, train_labels, test_data, test_labels, val_data, val_labels, xtrain_noisy, xtest_noisy, xval_noisy


def plot_images(images, labels, n_images=10, title='Images'):
    
    """
    Function to plot images

    Parameters
    ----------
    images : torch.Tensor
        images to plot
    labels : torch.Tensor
        labels of the images
    n_images : int
        number of images to plot. Default is 10
    title : str
        title of the plot. Default is 'Images'
    """

    # Plot the first 10 images of the training set
    for i, (image, label) in enumerate(zip(images, labels)):
        if i == n_images:
            break
        plt.figure(figsize=(10, 10))
        plt.subplot(1, n_images, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title(label.item())

    plt.suptitle(title)
    plt.show()



# function for getting the data of a given dataset
def build_data_2(train_loader, test_loader, n_classes: int, batch_size=10, noise_level=0.9, val_prop=0.1):
    
    """
    Function to get the data from a dataset

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        training set
    test_loader : torch.utils.data.DataLoader
        test set
    n_classes : int
        number of classes in the dataset
    batch_size : int
        batch size for the training and test set
    noise_level : float
        level of noise to add to the data. Default is 0.9
    val_prop : float
        proportion of the validation set with respect to
        the training set. Default is 0.1.

    Returns
    -------
    train_data : torch.Tensor
        training data
    train_labels : torch.Tensor
        training labels
    test_data : torch.Tensor
        test data
    test_labels : torch.Tensor
        test labels
    val_data : torch.Tensor
        validation data
    val_labels : torch.Tensor
        validation labels
    xtrain_noisy : torch.Tensor
        noisy training data
    xtest_noisy : torch.Tensor
        noisy test data
    xval_noisy : torch.Tensor
        noisy validation data
    """

    ### TEST DATA ###
    test_data = []
    test_labels = []

    # Collect all data and labels from the dataset
    for images, label in test_loader:
        test_data.append(images)
        test_labels.append(label)

    # Stack data and labels into tensors
    test_data = torch.cat(test_data, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    ### TRAINING DATA ###
    data = []
    labels = []

    # Collect all data and labels from the dataset
    for images, label in train_loader:
        data.append(images)
        labels.append(label)

    # Stack data and labels into tensors
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)

    # Convert tensors to NumPy arrays
    data_np = data.numpy()
    labels_np = labels.numpy()

    # Split the data into train and validation sets using train_test_split
    train_data_np, val_data_np, train_labels_np, val_labels_np = train_test_split(
        data_np, labels_np, test_size=val_prop, random_state=42)

    # Convert the NumPy arrays back to PyTorch tensors
    train_data = torch.from_numpy(train_data_np)
    val_data = torch.from_numpy(val_data_np)
    train_labels = torch.from_numpy(train_labels_np)
    val_labels = torch.from_numpy(val_labels_np)

    # Add noise to the training data
    xtrain_noisy = add_noise(train_data, noise_level)
    xtest_noisy = add_noise(test_data, noise_level)
    xval_noisy = add_noise(val_data, noise_level)

    # Print the sizes of each data set
    print(f'xtrain: {train_data.size()}')
    print(f'ytrain: {train_labels.size()}')

    print(f'\nxtest: {test_data.size()}')
    print(f'ytest: {test_labels.size()}')

    print(f'\nxval: {val_data.size()}')
    print(f'yval: {val_labels.size()}')
    
    print(f'\nxtrain noisy: {xtrain_noisy.size()}')
    print(f'xtest_noisy: {xtest_noisy.size()}')
    print(f'xval_noisy: {xval_noisy.size()}')

    return train_data, train_labels, test_data, test_labels, val_data, val_labels, xtrain_noisy, xtest_noisy, xval_noisy



""" MODELS """

class CNN():
    def __init__(self, n_output=10, channels=1, batch_size=10):

        """
        Class for the CNN model

        Parameters
        ----------
        n_output : int
            number of output classes. Default is 10
        channels : int
            number of channels of the input images. Default is 1
        batch_size : int
            batch size for the training and test set
        """

        self.model = nn.Sequential(
          
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, n_output),
        ) 
      
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss() 
        self.channels = channels
        self.batch_size = batch_size

    def testing(self, test_x, test_y):
        with torch.no_grad():
            output = self.model(test_x)
        
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        return accuracy_score(test_y.cpu(), predictions)*100
            
    def training(self, epochs: int, x_train, y_train, x_val, y_val, x_test, y_test):
        self.model.train()
  
        x_train, y_train = Variable(x_train.reshape(-1, self.batch_size, self.channels, 28, 28)), Variable(y_train.reshape(-1, self.batch_size))
        x_val, y_val = Variable(x_val.reshape(-1, self.channels, 28, 28)), Variable(y_val.reshape(-1))
        x_test, y_test = Variable(x_test.reshape(-1, self.channels, 28, 28)), Variable(y_test.reshape(-1))
      
        # run
        for epoch in range(epochs):
            for x, y, in zip(x_train, y_train):
                self.optimizer.zero_grad()
                
                output_train = self.model(x) 

                loss_train = self.criterion(output_train, y)

                loss_train.backward()
                self.optimizer.step()

            if epoch%2 == 0:
                clear_output(wait=True)
                print(f'Epoch: {epoch+1}\tloss: {loss_train.cpu().item():.3f}\tacc: {self.testing(x_val[:min(1000, len(x_val))], y_val[:min(1000, len(y_val))]):.2f}%')

        print('\n+training ended+')

        print('\n', '-'*50)
        print(f'\ntest accuracy: {self.testing(x_test[:min(1000, len(x_val))], y_test[:min(1000, len(x_val))]):.2f}%')


### DEEP CONVOLUTIONAL VAE ###

class Encoder(nn.Module):
    def __init__(self, d_in=784, z=36, channels=1, n_classes=10):
        super().__init__()

        self.z = z*2
        self.input_dim = d_in
        self.n_classes = n_classes

        # architecture

        self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=50)
        self.lin0 = nn.Linear(in_features=50, out_features=784)

        self.conv1 = nn.Conv2d(channels, 32, 4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.mpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.lrelu2 = nn.LeakyReLU(inplace=True)
        self.mpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 32, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.lrelu3 = nn.LeakyReLU(inplace=True)
        
        self.lin1 = nn.Linear(in_features=32*4*4, out_features=self.z)

    def forward(self, x, y):

        y = self.embed(y)
        y = self.lin0(y).reshape(-1, 1, 28, 28)
        
        # concatenation image + label
        x = torch.cat((x, y), axis=1)

        x = self.conv1(x)
        x = self.lrelu1(self.bn1(x))
        x = self.mpool1(x)

        x = self.conv2(x)
        x = self.lrelu2(self.bn2(x))
        x = self.mpool2(x)

        x = self.conv3(x)
        x = self.lrelu3(self.bn3(x))

        x = x.view(-1, 32*4*4)

        return self.lin1(x)

class Decoder(nn.Module):
    def __init__(self, d_in=784, channels=2, z=36, n_classes=10):
        super().__init__()

        self.z = z
        self.zsqrt = np.sqrt(z).astype(int)
        self.input_dim = d_in
        self.n_classes = n_classes
        self.channels = channels

        # architecture

        self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=50)
        self.lin0 = nn.Linear(in_features=50, out_features=z)
        self.lin1 = nn.Linear(in_features=z*2, out_features=2*10*10)
      
        self.tconv1 = nn.ConvTranspose2d(2, 64, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(inplace=True)

        self.tconv2 = nn.ConvTranspose2d(64, 64, 4, stride=1)
        self.lrelu2 = nn.LeakyReLU(inplace=True)

        self.tconv3 = nn.ConvTranspose2d(64, channels-1, 4, stride=2)

        self.sig = nn.Sigmoid()

    def forward(self, z, y):

        y = self.embed(y)
        y = self.lin0(y).reshape(-1, 1, self.zsqrt, self.zsqrt)

        # concatenation latent vector z + label
        x = torch.cat((z, y), axis=1).view(-1, self.channels*self.zsqrt*self.zsqrt)
        
        x = self.lin1(x).reshape(-1, self.channels, 10, 10)

        x = self.lrelu1(self.tconv1(x))
        x = self.lrelu2(self.tconv2(x))

        return self.sig(self.tconv3(x))

class DCCVAE(nn.Module):
    def __init__(self, d_in=784, z=36, n_classes=10, channels=2):
        super().__init__()
        
        self.z = z
        self.zsqrt = np.sqrt(z).astype(int)
        self.input_dim = d_in 

        #Encoder
        self.encoder = Encoder(d_in=d_in, z=z, channels=channels, n_classes=n_classes)
       
        #Decoder
        self.decoder = Decoder(d_in=d_in, z=z, channels=channels, n_classes=n_classes)

    def reparameterise(self, mu, logvar):  # mu + sigma * eps
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y):
        mu_logvar = self.encoder(x, y).view(-1, 2, self.z)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar).reshape(-1, 1, self.zsqrt, self.zsqrt)
        return self.decoder(z, y), mu, logvar

class DCCVAE_model:
    def __init__(self, n_input=784, n_classes=10, z=36, channels=2):

        self.n_input = n_input
        self.n_output = n_classes
        self.channels = channels

        self.cvae = DCCVAE(d_in=n_input, z=z, n_classes=n_classes, channels=channels)

        self.optimizer = optim.Adam(self.cvae.parameters(), lr=0.001)

        print(self.cvae)

    def loss_function(self, x_hat, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        return BCE + KLD

    def test(self, X, Y):
        with no_grad():
            loss = 0
            for x, y in zip(X, Y):
                x_hat, mu, logvar = self.cvae(x, y)
                loss += self.loss_function(x_hat, x, mu, logvar).cpu().numpy().item()
                
        return loss / len(X) 

    def training(self, epochs: int, xtrain, ytrain, xval, yval):
        train_losses = []
        test_losses = []

        xtrain = Variable(xtrain)
        ytrain = Variable(ytrain)

        xval = Variable(xval)
        yval = Variable(yval)

        self.cvae.train()
        for epoch in range(1, epochs+1):
            train_loss = 0
            
            for x, y in zip(xtrain, ytrain):
                
                x_hat, mu, logvar = self.cvae(x, y)
                loss = self.loss_function(x_hat, x, mu, logvar)
                train_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            test_loss = self.test(xval, yval)

            train_losses.append(train_loss/len(xtrain))
            test_losses.append(test_loss)
        
            # plotting
            clear_output(wait=True)
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
            ax1.plot(range(epoch), train_losses, label=f'training loss {train_losses[-1]:.2f}')
            ax1.plot(range(epoch), test_losses, label=f'test loss {test_loss:.2f}')

            ax1.set_xlim((0, epochs))
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('loss')
            ax1.set_title(f'Epoch {epoch}|{epochs} - Loss')
            ax1.legend(loc='upper right')
            
            yu = ytest[np.random.randint(self.n_output), 0]
            img = self.cvae.decoder(torch.randn(1, 1, self.cvae.zsqrt, self.cvae.zsqrt), 
                                    yu).detach().cpu().numpy()
            img = img.reshape(28, 28) if self.channels == 2 else img.reshape(28, 28, 3)
            ax2.imshow(img, cmap='Greys')
            ax2.set_xticks(())
            ax2.set_yticks(())
            ax2.set_title(f'generated sample: {yu}')
            plt.pause(0.0001)
        
        print('\n- training ended -')

    def generate(self, n_classes=2):
        fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=5)
        with no_grad():
            for row in range(2):
                for ax in axs[row]:
                    idx = torch.randint(n_classes, (1, 1))
                    img = self.cvae.decoder(torch.randn(1, 1, self.cvae.zsqrt, self.cvae.zsqrt), 
                                    idx).detach().cpu().numpy()
                    img = img.reshape(28, 28) if self.channels == 2 else img.reshape(28, 28, 3)
                    ax.imshow(img, cmap='Greys')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title(f'class {idx.item()}')
                    
        plt.show()

    def generate_one(self, n_classes, idx):
        with no_grad():
            idx = torch.tensor([[idx]])
            
            img = self.cvae.decoder(torch.randn(1, 1, self.cvae.zsqrt, self.cvae.zsqrt), 
                                    idx).detach().cpu().numpy()
            img = img.reshape(28, 28) if self.channels == 2 else img.reshape(28, 28, 3)
            return img, idx


### CONDITIONAL GAN ###

class CGAN:
    def __init__(self, n_input=784, n_output=10, nz=20, channels=1):
        
        self.n_in = n_input
        self.n_out = n_output
        self.nz = nz
        self.nc = channels

        # networks
        self.generator = nn.Sequential(
            nn.Linear(in_features=nz + n_output, out_features=300),
            nn.ReLU(),

            nn.Linear(300, 400),
            nn.ReLU(),

            nn.Linear(400, n_input),
            nn.Sigmoid(),

        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=n_input + n_output, out_features=400),
            nn.ReLU(),

            nn.Linear(400, 300),
            nn.ReLU(),

            nn.Linear(300, 1),
            nn.Sigmoid(),
        )
        
        # setup optimizer
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()

        # labels
        self.real_label = 1
        self.fake_label = 0
        
    def training(self, epochs, xtrain, ytrain, batch_size=1, verbose=True):
    
        t0 = time.time()
        sec, mins = 0, 0

        loss_Dis = []
        loss_Gen = []
        acc_real = []
        acc_fake = []

        for epoch in range(epochs):
            for i, (real_img, y) in enumerate(zip(xtrain, ytrain)):

                ## training the Discriminator ##
                self.discriminator.zero_grad()

                # real images
                real_cpu = real_img
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), self.real_label,
                                   dtype=real_cpu.dtype)  # label = (1, 1, ... 1)

                # forward 1
                output = self.discriminator(torch.cat((real_cpu.view(-1, self.n_in), y), axis=1)).view(-1, 1).squeeze(1)  # forward D
                errD_real = self.criterion(output, label) # loss 4 real img
                errD_real.backward()
                D_x = output.mean().item()  # D(x)

                # fake images
                noise = torch.randn(batch_size, self.nz)
                fake = self.generator(torch.cat((noise, y), axis=1))  
                label.fill_(self.fake_label)

                # forward 2
                output = self.discriminator(torch.cat((fake.detach(), y), axis=1)).view(-1, 1).squeeze(1)  
                errD_fake = self.criterion(output, label)  
                errD_fake.backward()
                D_G_z1 = output.mean().item()  # D(G(z))

                errD = errD_real + errD_fake  # total loss for D
                self.optimizerD.step()


                ## training the Generator ##

                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost

                output = self.discriminator(torch.cat((fake, y), axis=1)).view(-1, 1).squeeze(1)  
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()


                ## display ##
                if i % 400 == 0 and verbose:
                    loss_Dis.append(errD.item())
                    loss_Gen.append(errG.item())
                    acc_real.append(D_x)
                    acc_fake.append(D_G_z1)

                    # plotting
                    clear_output(wait=True)
                    plt.clf()

                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
                    plt.tight_layout(h_pad=3)

                    # losses
                    ax1.plot(range(len(loss_Dis)), loss_Dis, label=f'Discriminator loss: {loss_Dis[-1]:.2f}')
                    ax1.plot(range(len(loss_Gen)), loss_Gen, label=f'Generator loss: {loss_Gen[-1]:.2f}')

                    ax1.set_ylabel('Loss')
                    ax1.set_title(f'Loss (Epoch {epoch+1}/{epochs})')
                    ax1.legend(loc='upper right')

                    # accuracy
                    ax2.plot(range(len(acc_real)), acc_real, label=f'Accuracy for Real (1): {acc_real[-1]:.3f}')
                    ax2.plot(range(len(acc_fake)), acc_fake, label=f'Accuracy for Fake (0): {acc_fake[-1]:.3f}')

                    ax2.set_ylim((-0.1, 1.1))
                    ax2.set_ylabel('Accuracy')
                    ax2.set_title(f'Accuracy (Epoch {epoch+1}/{epochs})')
                    ax2.legend(loc='upper right')

                    # novel images
                    targ = y[np.random.randint(0, len(y))]
                    img = self.generator(torch.cat((torch.randn(1, self.nz),
                                                     targ.reshape(1, self.n_out)), axis=1
                                                    )).detach().numpy()

                    ax3.imshow(img.reshape(28, 28), cmap='Greys')
                    ax3.set_xticks(())
                    ax3.set_yticks(())
                    ax3.set_title(f'Generated sample: {torch.argmax(targ).item()}')

                    display(fig)
                    plt.close(fig)


            if epoch == 0:
                dft0 = time.time() - t0
            dft = dft0 * (epochs - epoch)
            sec = dft % 60
            mins = (dft - sec) // 60

        print('\n- training ended -')
        
    def training2(self, epochs, xtrain, ytrain, batch_size=1, verbose=True):
        
        t0 = time.time()
        sec, mins = 0, 0

        loss_Dis = []
        loss_Gen = []
        acc_real = []
        acc_fake = []
        
        for epoch in range(epochs):
            for i, real_img, y in zip(range(len(xtrain)), xtrain, ytrain):

                ## training the Discriminator ##
                self.discriminator.zero_grad()
                
                # real images
                real_cpu = real_img
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), self.real_label,
                                   dtype=real_cpu.dtype)  # label = (1, 1, ... 1)

                # forward 1
                output = self.discriminator(torch.cat((real_cpu.view(-1, self.n_in), y), axis=1)).view(-1, 1).squeeze(1)  # forward D
                errD_real = self.criterion(output, label) # loss 4 real img
                errD_real.backward()
                D_x = output.mean().item()  # D(x)

                # fake images
                noise = torch.randn(batch_size, self.nz)
                fake = self.generator(torch.cat((noise, y), axis=1))  
                label.fill_(self.fake_label)
                
                # forward 2
                output = self.discriminator(torch.cat((fake.detach(), y), axis=1)).view(-1, 1).squeeze(1)  
                errD_fake = self.criterion(output, label)  
                errD_fake.backward()
                D_G_z1 = output.mean().item()  # D(G(z))

                errD = errD_real + errD_fake  # total loss for D
                self.optimizerD.step()


                ## training the Generator ##

                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost

                output = self.discriminator(torch.cat((fake, y), axis=1)).view(-1, 1).squeeze(1)  
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()


                ## display ##

                if i%400 == 0 and verbose:
                    loss_Dis.append(errD.item())
                    loss_Gen.append(errG.item())
                    acc_real.append(D_x)
                    acc_fake.append(D_G_z1)

                
                    # plotting
                    clear_output(wait=True)
                    plt.clf()

                    fig = plt.figure(figsize=(16, 4))
                    ax1 = plt.subplot(2, 2, 1)
                    ax2 = plt.subplot(2, 2, 3)
                    ax3 = plt.subplot(1, 2, 2)
                    plt.tight_layout(h_pad=3)

                    # losses
                    ax1.plot(range(len(loss_Dis)), loss_Dis, label=f'Discrimator loss: {loss_Dis[-1]:.2f}')
                    ax1.plot(range(len(loss_Gen)), loss_Gen, label=f'Generator loss:   {loss_Gen[-1]:.2f}')

                    #ax1.set_xlim((0, epochs*len(train_data)))
                    #ax1.set_xlabel('epochs')
                    ax1.set_ylabel('loss')
                    ax1.set_title(f'Loss | Epoch {epoch+1} of {epochs}')
                    ax1.legend(loc='upper right')

                    # accuracy
                    ax2.plot(range(len(acc_real)), acc_real, label=f'accuracy for Real (1): {acc_real[-1]:.3f}')
                    ax2.plot(range(len(acc_fake)), acc_fake, label=f'accuracy for Fake (0): {acc_fake[-1]:.3f}')

                    ax2.set_ylim((-0.1, 1.1))
                    #ax2.set_xlim((0, epochs*len(train_data)))
                    #ax2.set_xlabel('epochs')
                    ax2.set_ylabel('accuracy')
                    ax2.set_title(f'Accuracy | remaining: {mins:.0f}m {sec:.0f}s')
                    ax2.legend(loc='upper right')
                    
                    # novel images
                    #print(y.shape)
                    targ = y[np.random.randint(0, len(y))]
                    img = self.generator(torch.cat((torch.randn(1, self.nz), 
                                                    targ.reshape(1, self.n_out)), axis=1
                                                   )).detach().numpy().reshape(28, 28)
                    
                    ax3.imshow(img, cmap='Greys')
                    ax3.set_xticks(())
                    ax3.set_yticks(())
                    ax3.set_title(f'Generated sample: {torch.argmax(targ).item()} {img.shape}')

                    plt.pause(0.0001)
                
            if epoch == 0:
                dft0 = time.time() - t0
            dft = dft0 * (epochs - epoch)
            sec = dft%60
            mins = (dft-sec)//60

        print('\n- training ended -')

    def generate(self):
        fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=5)
        with no_grad():
            idx = 0
            for row in range(2):
                for ax in axs[row]:
                    img = self.generator(torch.cat((torch.randn(1, self.nz), 
                                                    hot_numbers[idx].reshape(1, self.n_out)), axis=1
                                                   )).detach().numpy().reshape(28, 28)
                    ax.imshow(img, cmap='Greys')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    #ax.set_title(torch.argmax(hot_numbers[idx]).item())
                    idx += 1
        plt.show()

    def generate2(self, n_classes=2):
        fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=5)
        with no_grad():
            for row in range(2):
                for ax in axs[row]:
                    idx = np.random.randint(0, n_classes)
                    hot = torch.zeros(n_classes)
                    hot[idx] = 1
                    img = self.generator(torch.cat((torch.rand(1, self.nz), 
                                                      hot.reshape(1, n_classes)), 
                                                    axis=1)).numpy().reshape(28, 28)
                    ax.imshow(img, cmap='Greys')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title(f'class {idx}')
                    
        plt.show()


### DEEP CONVOLUTIONAL CONDITIONAL GAN ###

class Discriminator(nn.Module):
    def __init__(self, d_in=784, channels=2, n_classes=10):
        super().__init__()

        self.input_dim = d_in
        self.n_classes = n_classes
        self.channels = channels

        # architecture
        self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=20)
        self.lin0 = nn.Linear(in_features=20, out_features=1*28*28)

        self.conv1 = nn.Conv2d(channels, 32, 4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.mpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.lrelu2 = nn.LeakyReLU(inplace=True)
        self.mpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 32, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.lrelu3 = nn.LeakyReLU(inplace=True)
        
        self.lin1 = nn.Linear(in_features=32*4*4, out_features=1)
        self.sig = nn.Sigmoid() 

    def forward(self, x, y):

        y = self.embed(y)
        y = self.lin0(y).reshape(-1, 1, 28, 28)

        # concatenation image + label
        x = torch.cat((x, y), axis=1)

        x = self.conv1(x)
        x = self.lrelu1(self.bn1(x))
        x = self.mpool1(x)

        x = self.conv2(x)
        x = self.lrelu2(self.bn2(x))
        x = self.mpool2(x)

        x = self.conv3(x)
        x = self.lrelu3(self.bn3(x))

        x = x.view(-1, 32*4*4)

        return self.sig(self.lin1(x))

class Generator(nn.Module):
    def __init__(self, d_in=784, channels=2, z=36, n_classes=10):
        super().__init__()

        self.z = z
        self.zsqrt = np.sqrt(z).astype(int)
        self.input_dim = d_in
        self.n_classes = n_classes
        self.channels = channels

        # architecture
        self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=20)
        self.lin0 = nn.Linear(in_features=20, out_features=z)
        self.lin1 = nn.Linear(in_features=z*self.channels, out_features=self.channels*10*10)
      
        self.tconv1 = nn.ConvTranspose2d(channels, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lrelu1 = nn.LeakyReLU(inplace=True)

        self.tconv2 = nn.ConvTranspose2d(64, 64, 4, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lrelu2 = nn.LeakyReLU(inplace=True)

        self.tconv3 = nn.ConvTranspose2d(64, self.channels-1, 4, stride=2)

        self.sig = nn.Sigmoid()
        

    def forward(self, z, y):

        y = self.embed(y)
        y = self.lin0(y).reshape(-1, 1, self.zsqrt, self.zsqrt)

        # concatenation latent vector z + label
        x = torch.cat((z, y), axis=1).view(-1, self.channels*self.zsqrt*self.zsqrt)

        x = self.lin1(x).reshape(-1, self.channels, 10, 10)

        x = self.bn1(self.tconv1(x))
        x = self.lrelu1(x)

        x = self.bn2(self.tconv2(x))
        x = self.lrelu2(x)

        return self.sig(self.tconv3(x))

class DCCGAN:
    def __init__(self, n_input=784, n_classes=10, z=20, channels=1):
        
        self.n_in = n_input
        self.n_out = n_classes
        self.z = z
        self.zsqrt = np.sqrt(z).astype(int)
        self.channels = channels

        # networks
        self.generator = Generator(d_in=n_input, channels=channels, z=z, n_classes=n_classes)
        print(self.generator)
        
        self.discriminator = Discriminator(d_in=n_input, channels=channels, n_classes=n_classes)
        print(self.discriminator)

        # setup optimizer
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()

        # labels
        self.real_label = 1
        self.fake_label = 0
        
    def training(self, epochs, xtrain, ytrain, xval, yval, verbose=True):
        
        xtrain = Variable(xtrain)
        ytrain = Variable(ytrain)
        
        xval = Variable(xval)
        yval = Variable(yval)

        # for time
        t0 = time.time()
        sec, mins = 0, 0

        loss_Dis = []
        loss_Gen = []
        acc_real = []
        acc_fake = []
        
        for epoch in range(epochs):
            for i, real_img, y in zip(range(len(xtrain)), xtrain, ytrain):

                ## training the Discriminator ##
                self.discriminator.zero_grad()
                
                # real images
                batch_size = real_img.size(0)
                label = torch.full((batch_size,), self.real_label,
                                   dtype=real_img.dtype) # label = (1, 1, ... 1)

                # forward 1
                output = self.discriminator(real_img, y).view(-1, 1).squeeze(1)  # forward D
                errD_real = self.criterion(output, label) # loss 4 real img
                errD_real.backward()
                D_x = output.mean().item()  # D(x)

                # fake images
                noise = torch.randn(batch_size, self.channels-1, self.zsqrt, self.zsqrt)
                fake = self.generator(noise, y) 

                label.fill_(self.fake_label)
                
                # forward 2
                
                output = self.discriminator(fake.detach(), y).view(-1, 1).squeeze(1)  
                errD_fake = self.criterion(output, label)  
                errD_fake.backward()
                D_G_z1 = output.mean().item()  # D(G(z))

                errD = errD_real + errD_fake  # total loss for D
                self.optimizerD.step()


                ## training the Generator ##

                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost

                output = self.discriminator(fake, y).view(-1, 1).squeeze(1)  
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()


                ## display ##

                if i%400 == 0 and verbose:
                    loss_Dis.append(errD.item())
                    loss_Gen.append(errG.item())
                    acc_real.append(D_x)
                    acc_fake.append(D_G_z1)

                
                    # plotting
                    clear_output(wait=True)
                    plt.clf()

                    fig = plt.figure(figsize=(16, 4))
                    ax1 = plt.subplot(2, 2, 1)
                    ax2 = plt.subplot(2, 2, 3)
                    ax3 = plt.subplot(1, 2, 2)
                    plt.tight_layout(h_pad=3)

                    # losses
                    ax1.plot(range(len(loss_Dis)), loss_Dis, label=f'Discrimator loss: {loss_Dis[-1]:.2f}')
                    ax1.plot(range(len(loss_Gen)), loss_Gen, label=f'Generator loss:   {loss_Gen[-1]:.2f}')

                    ax1.set_ylabel('loss')
                    ax1.set_title(f'Loss | Epoch {epoch+1} of {epochs}')
                    ax1.legend(loc='upper right')

                    # accuracy
                    ax2.plot(range(len(acc_real)), acc_real, label=f'classification of Real (1): {acc_real[-1]:.3f}')
                    ax2.plot(range(len(acc_fake)), acc_fake, label=f'classification of Fake (0): {acc_fake[-1]:.3f}')

                    ax2.set_ylim((-0.1, 1.1))
                    ax2.set_ylabel('accuracy')
                    ax2.set_title(f'Accuracy | remaining: {mins:.0f}m {sec:.0f}s')
                    ax2.legend(loc='upper right')
                    
                    # novel images
                    targ = ytrain[np.random.randint(len(ytrain)), 0]
                    img = self.generator(torch.randn(1, self.channels-1, self.zsqrt, self.zsqrt), 
                                                    targ).detach().cpu().numpy()
                    img = img.reshape(28, 28) if self.channels == 2 else img.reshape(28, 28, 3)
                    ax3.imshow(img, cmap='Greys')
                    ax3.set_xticks(())
                    ax3.set_yticks(())
                    ax3.set_title(f'Generated sample: {targ.item()}')

                    plt.pause(0.0001)
                
            if epoch == 0:
                dft0 = time.time() - t0
            dft = dft0 * (epochs - epoch)
            sec = dft%60
            mins = (dft-sec)//60

        print('\n- training ended -')

    def generate(self, n_classes=2):
        fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=5)
        with no_grad():
            for row in range(2):
                for ax in axs[row]:
                    idx = torch.randint(n_classes, (1, 1))
                 
                    img = self.generator(torch.randn(1, self.channels-1, self.zsqrt, self.zsqrt), 
                                    idx).detach().cpu().numpy()
                    img = img.reshape(28, 28) if self.channels == 2 else img.reshape(28, 28, 3)
                    ax.imshow(img, cmap='Greys')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title(f'class {idx.item()}')
                    
        plt.show()

    def generate_one(self, n_classes, idx):
        with no_grad():
            idx = torch.tensor([[idx]])
            
            img = self.generator(torch.randn(1, self.channels-1, self.zsqrt, self.zsqrt), 
                            idx).detach().cpu().numpy()
            img = img.reshape(28, 28) if self.channels == 2 else img.reshape(28, 28, 3)
            return img, idx


### DENOISING AUTOENCODER ###

class DAE:
    def __init__(self, n_input=784, channels=1):

        self.n_input = n_input
        self.channels = channels

        self.dae = nn.Sequential(
            nn.Conv2d(channels, 64, 4, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 4, 2, padding=1),
            nn.BatchNorm2d(num_features=4),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(4, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 4, stride=1),
            nn.LeakyReLU(64),

            nn.ConvTranspose2d(64, channels, 4, stride=2),
            nn.Sigmoid(),
        )

        self.optimizer = optim.Adam(self.dae.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=0.001)
        print(self.dae)

    def test(self, X, Xn):
        with no_grad():
            loss = 0
            for x, x_noisy in zip(X, Xn):
                x_hat = self.dae(x_noisy)
                loss += nn.functional.mse_loss(x_hat, x, reduction='sum')
                
        return loss / len(X)

    def training(self, epochs, xtrain, xtrain_noisy, xtest, xtest_noisy, verbose=True):
        train_losses = []
        test_losses = []

        xtrain = Variable(xtrain)
        xtrain_noisy = Variable(xtrain_noisy).float()

        xtest = Variable(xtest)
        xtest_noisy = Variable(xtest_noisy).float()

        self.dae.train()
        for epoch in range(1, epochs+1):
            train_loss = 0
            
            for i, x, x_noisy in zip(range(len(xtrain)), xtrain, xtrain_noisy):
              
                x_hat = self.dae(x_noisy)
                loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
                train_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                # plotting
                if i%400 == 0 and verbose:
                    test_loss = self.test(xtest, xtest_noisy)

                    train_losses.append(train_loss/len(xtrain))
                    test_losses.append(test_loss)
                
                    clear_output(wait=True)

                    ###########################################################
                    plt.clf()
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(20, 4), ncols=4)
                    ax1.plot(range(len(train_losses)), train_losses, 
                             label=f'training loss {train_losses[-1]:.2f}')
                    ax1.plot(range(len(test_losses)), test_losses, 
                             label=f'test loss {test_loss:.2f}')

                    ax1.set_ylabel('loss')
                    ax1.set_title(f'Epoch {epoch}|{epochs} - Loss')
                    ax1.legend(loc='upper right')
                    
                    idx = np.random.randint(0, len(x))

                    img_x = x[idx, ...].cpu().numpy()
                    img_x = img_x.reshape(28, 28) if self.channels == 1 else img_x.reshape(28, 28, 3)
                    ax2.imshow(img_x, cmap='Greys')
                    ax2.set_xticks(())
                    ax2.set_yticks(())
                    ax2.set_title('target image')

                    img_noisy = x_noisy[idx, ...].cpu().numpy()
                    img_noisy = img_noisy.reshape(28, 28) if self.channels == 1 else img_noisy.reshape(28, 28, 3)
                    ax3.imshow(img_noisy, cmap='Greys')
                    ax3.set_xticks(())
                    ax3.set_yticks(())
                    ax3.set_title('noisy image')
                    
                    img_hat = self.dae(x_noisy[idx, ...].reshape(1, self.channels, 28, 28)).detach().cpu().numpy()
                    img_hat = img_hat.reshape(28, 28) if self.channels == 1 else img_hat.reshape(28, 28, 3)

                    ax4.imshow(img_hat, cmap='Greys')
                    ax4.set_xticks(())
                    ax4.set_yticks(())
                    ax4.set_title('reconstructed image')
                    
                    plt.pause(0.0001)
            
            if not verbose:
                clear_output(wait=True)
                print(f'epoch: {epoch}|{epochs}')

        print('\n- training ended -')

    def reconstruct(self):

        for i, x, x_noisy in zip(range(len(xtest)), xtest, xtest_noisy):

            clear_output(wait=True)
            
            plt.clf()
            fig, (ax2, ax3, ax4) = plt.subplots(figsize=(17, 4), ncols=3)
            idx = np.random.randint(0, len(x))

            img_x = x[idx, ...].cpu().numpy()
            img_x = img_x.reshape(28, 28) if self.channels == 1 else img_x.reshape(28, 28, 3)
            ax2.imshow(img_x, cmap='Greys')
            ax2.set_xticks(())
            ax2.set_yticks(())
            ax2.set_title('target image')

            img_noisy = x_noisy[idx, ...].cpu().numpy()
            img_noisy = img_noisy.reshape(28, 28) if self.channels == 1 else img_noisy.reshape(28, 28, 3)
            ax3.imshow(img_noisy, cmap='Greys')
            ax3.set_xticks(())
            ax3.set_yticks(())
            ax3.set_title('noisy image')
            
            img_hat = self.dae(x[idx, ...].reshape(1, self.channels, 28, 28)).detach().cpu().numpy()
            img_hat = img_hat.reshape(28, 28) if self.channels == 1 else img_hat.reshape(28, 28, 3)

            ax4.imshow(img_hat, cmap='Greys')
            ax4.set_xticks(())
            ax4.set_yticks(())
            ax4.set_title('reconstructed image')
            
            plt.pause(0.3)

            if i == 100:
                break
