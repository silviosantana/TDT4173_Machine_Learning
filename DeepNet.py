# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:17:03 2019

@author: silvi
"""
import numpy as np
import util
import torch

import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms.functional import to_tensor, normalize
from torch import nn
#from utils import to_cuda, compute_loss_and_accuracy

def dataloader(X, Y, batch_size):
    X = X.reshape((len(X), 1, 20, 20))
    #X_train, Y_train, X_test, Y_test = util.train_val_split(X, Y, 0.2)
    
    #X_train, Y_train, X_val, Y_val = util.train_val_split(X_train, Y_train, 0.1)
    
    X_train = X
    Y_train = Y
    #my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
    #my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

    tensor_x = torch.stack([torch.Tensor(i) for i in X_train]) 
    tensor_y = torch.stack([torch.Tensor(i) for i in Y_train])
    
    dataset_train = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    #tensor_x = torch.stack([torch.Tensor(i) for i in X_test]) 
    #tensor_y = torch.stack([torch.Tensor(i) for i in Y_test])
    
    dataset_test = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    #tensor_x = torch.stack([torch.Tensor(i) for i in X_val]) 
    #tensor_y = torch.stack([torch.Tensor(i) for i in Y_val])
    
    dataset_val = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=batch_size,
                                                       num_workers=2)
    
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 num_workers=2)
    
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
    
    return dataloader_train, dataloader_val, dataloader_test

def to_cuda(elements):
    """
    Transfers elements to GPU memory, if a nvidia- GPU is available.
    Args:
        elements: A list or a single pytorch module.
    Returns:
        The same list transferred to GPU memory
    """

    if torch.cuda.is_available(): # Checks if a GPU is available for pytorch
        if isinstance(elements, (list, tuple)):
            return [x.cuda() for x in elements] # Transfer each index of the list to GPU memory
        return elements.cuda()
    return elements


def compute_loss_and_accuracy(dataloader, model, loss_criterion):
    """
    Computes the total loss and accuracy over the whole dataloader
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: nn.CrossEntropyLoss()
    Returns:
        [loss_avg, accuracy]: both scalar.
    """
    # Tracking variables
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0

    for (X_batch, Y_batch) in dataloader:
        # Transfer images/labels to GPU VRAM, if possible
        print(X_batch.shape)
        print(Y_batch.shape)
        X_batch = to_cuda(X_batch)
        Y_batch = to_cuda(Y_batch)
        # Forward pass the images through our model
        output_probs = model(X_batch)
        # Compute loss
        Y_batch = Y_batch.long()
        print(output_probs.shape)
        loss = loss_criterion(output_probs, Y_batch)
        #loss = loss_criterion(torch.max(output_probs, 1)[1], torch.max(Y_batch, 1)[1])
        

        # Predicted class is the max index over the column dimension
        predictions = output_probs.argmax(dim=1).squeeze()
        Y_batch = Y_batch.squeeze()

        # Update tracking variables
        loss_avg += loss.item()
        total_steps += 1
        total_correct += (predictions == Y_batch).sum().item()
        total_images += predictions.shape[0]
    loss_avg = loss_avg / total_steps
    accuracy = total_correct / total_images
    return loss_avg, accuracy


class DeepCNNModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*1*1
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x

def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class Trainer:

    def __init__(self, X, Y, batch_size):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 2
        self.batch_size = batch_size
        self.learning_rate = 5e-4
        self.early_stop_count = 4


        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = DeepCNNModel(image_channels=1, num_classes=26)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloader(X, Y, batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all three datasets.
                if batch_it % self.validation_check == 0:
                    self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        return
    



if __name__ == "__main__":
    images, labels = util.import_dataset()
    X = np.array([np.array(i) for i in images])
    Y = np.array([np.array(i) for i in util.onehot_encode(labels)])
    #Y = [np.array(i) for i in labels]
    #Y = np.array(Y).reshape(len(Y),1)
    batch_size = 2
    
    print("Training ResNet18")
    trainer = Trainer(X, Y, batch_size)
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss_ResNet18.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy_ResNet18.png"))
    plt.show()

    print("ResNet18 Results")
    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
    print("Final training accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])

    print("Final test loss:", trainer.TEST_LOSS[-trainer.early_stop_count])
    print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])
    print("Final training loss:", trainer.TRAIN_LOSS[-trainer.early_stop_count])

    #Training DeepCNN for comparison
    print("Training Deep CNN")
    trainer_cnn = Trainer(m = 1)
    trainer_cnn.train()

    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="[ResNet18] Validation")
    plt.plot(trainer.TRAIN_LOSS, label="[ResNet18] Training")
    plt.plot(trainer.TEST_LOSS, label="[ResNet18] Test")
    plt.plot(trainer_cnn.VALIDATION_LOSS, label="[DeepCNN] Validation")
    plt.plot(trainer_cnn.TRAIN_LOSS, label="[DeepCNN] Training")
    plt.plot(trainer_cnn.TEST_LOSS, label="[DeepCNN] Testing")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss_Comparison.png"))
    plt.show()

    print("Deep CNN Results")
    print("Final test accuracy:", trainer_cnn.TEST_ACC[-trainer_cnn.early_stop_count])
    print("Final validation accuracy:", trainer_cnn.VALIDATION_ACC[-trainer_cnn.early_stop_count])
    print("Final training accuracy:", trainer_cnn.TRAIN_ACC[-trainer_cnn.early_stop_count])

    print("Final test loss:", trainer_cnn.TEST_LOSS[-trainer_cnn.early_stop_count])
    print("Final validation loss:", trainer_cnn.VALIDATION_LOSS[-trainer_cnn.early_stop_count])
    print("Final training loss:", trainer_cnn.TRAIN_LOSS[-trainer_cnn.early_stop_count])

