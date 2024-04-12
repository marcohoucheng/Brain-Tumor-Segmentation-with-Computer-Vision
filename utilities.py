import os, random, time
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

# Pytorch functions
import torch
# Neural network layers
import torch.nn as nn
import torch.nn.functional as F
# Optimizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
# Torchvision library
from torchvision import transforms

# For results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y): # DICE
  '''
  Compute accuracy from ground-truth and predicted labels.

  Input
  ------
  y_pred: torch.Tensor [BATCH_SIZE, N_LABELS]
  y: torch.Tensor [BATCH_SIZE]

  Output
  ------
  acc: float
    Accuracy
  '''
  y_prob = torch.sigmoid(y_pred)
  y_pred = (y_prob>0.5).int()
  acc = 0
  for i in range(y.shape[0]):
     y_tmp = y[i,:,:]
     y_pred_tmp = y_pred[i,:,:]
     acc_tmp = y_tmp.eq(y_pred_tmp).sum()/np.prod(y_tmp.shape)
     acc += acc_tmp
  acc /= y.shape[0]
  return acc

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, iterator, optimizer, criterion, device):
  batch_loss = 0
  batch_acc = 0

  # Train mode
  model.train()

  batch_loss = []
  batch_acc = []
  
  for i, (x,y) in enumerate(iterator):

    x = x.to(device)
    y = y.to(device)

    # Set gradients to zero
    optimizer.zero_grad()

    # Make Predictions
    y_pred = model(x)
    y_pred = y_pred.squeeze(1)

    # Compute loss
    loss = criterion(y_pred, y.float())

    # Compute accuracy
    acc = calculate_accuracy(y_pred, y)

    # Backprop
    loss.backward()

    # Apply optimizer
    optimizer.step()

    # Extract data from loss and accuracy
    batch_loss.append(loss.item())
    batch_acc.append(acc.item())

    print("{0:0.1f}".format((i+1)/len(iterator)*100), "% loaded in this epoch for training", end="\r")
  print("\n")
  return np.sum(batch_loss)/len(iterator), np.sum(batch_acc)/len(iterator), batch_loss, batch_acc

def evaluate(model, iterator, criterion, device):
  batch_loss = 0
  batch_acc = 0

  # Evaluation mode
  model.eval()

  batch_loss = []
  batch_acc = []

  # Do not compute gradients
  with torch.no_grad():

    for i, (x,y) in enumerate(iterator):

      x = x.to(device)
      y = y.to(device)

      # Make Predictions
      y_pred = model(x)
      y_pred = y_pred.squeeze(1) # [B, 64, 64]

      # Compute loss
      loss = criterion(y_pred, y.float())

      # Compute accuracy
      acc = calculate_accuracy(y_pred, y)

      # Extract data from loss and accuracy
      batch_loss.append(loss.item())
      batch_acc.append(acc.item())

      print("{0:0.1f}".format((i+1)/len(iterator)*100), "% loaded in this epoch for evaluation.", end="\r")
  print("\n")
  return np.sum(batch_loss)/len(iterator), np.sum(batch_acc)/len(iterator), batch_loss, batch_acc

def model_training(n_epochs, model, train_iterator, valid_iterator, optimizer, criterion, device, model_name='best_model.pt'):

  # Initialize validation loss
  best_valid_loss = float('inf')

  # Save output losses, accs
  train_losses = []
  train_accs = []
  valid_losses = []
  valid_accs = []

  train_batch_losses = []
  train_batch_accs = []
  valid_batch_losses = []
  valid_batch_accs = []

  # Loop over epochs
  for epoch in range(n_epochs):
    start_time = time.time()
    # Train
    train_loss, train_acc, train_batch_loss, train_batch_acc = train(model, train_iterator, optimizer, criterion, device)
    # Validation
    valid_loss, valid_acc, valid_batch_loss, valid_batch_acc = evaluate(model, valid_iterator, criterion, device)
    # Save best model
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      # Save model
      torch.save(model.state_dict(), model_name)
    end_time = time.time()

    print("---------------------------------")
    print(f"\nEpoch: {epoch+1}/{n_epochs} -- Epoch Time: {end_time-start_time:.2f} s")
    print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_acc * 100:.2f}%")
    print(f"Val -- Loss: {valid_loss:.3f}, Acc: {valid_acc * 100:.2f}%")

    # Save
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

    train_batch_losses.append(train_batch_loss)
    train_batch_accs.append(train_batch_acc)
    valid_batch_losses.append(valid_batch_loss)
    valid_batch_accs.append(valid_batch_acc)

    # Early stopping
    early_stopper = EarlyStopper(patience=5, min_delta=0)
    if early_stopper.early_stop(valid_loss):             
      break

  return train_losses, train_accs, valid_losses, valid_accs, train_batch_losses, train_batch_accs, valid_batch_losses, valid_batch_accs

def plot_results(n_epochs, train_losses, train_accs, valid_losses, valid_accs):
  N_EPOCHS = n_epochs
  # Plot results
  plt.figure(figsize=(20, 6))
  _ = plt.subplot(1,2,1)
  plt.plot(np.arange(N_EPOCHS)+1, train_losses, linewidth=3)
  plt.plot(np.arange(N_EPOCHS)+1, valid_losses, linewidth=3)
  _ = plt.legend(['Train', 'Validation'])
  plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Loss')

  _ = plt.subplot(1,2,2)
  plt.plot(np.arange(N_EPOCHS)+1, train_accs, linewidth=3)
  plt.plot(np.arange(N_EPOCHS)+1, valid_accs, linewidth=3)
  _ = plt.legend(['Train', 'Validation'])
  plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')

def model_testing(model, test_iterator, criterion, device, model_name='best_model.pt'):
  # Test model
  model.load_state_dict(torch.load(model_name))
  test_loss, test_acc, test_batch_loss, test_batch_acc = evaluate(model, test_iterator, criterion, device)
  print(f"Test -- Loss: {test_loss:.3f}, Acc: {test_acc * 100:.2f} %")
  return test_loss, test_acc, test_batch_loss, test_batch_acc
  
def predict(model, iterator, device):

  # Evaluation mode
  model.eval()

  labels = []
  preds = []

  with torch.no_grad():
    for (x, y) in iterator:
      x = x.to(device)
      y = y.int().to(device)

      y_pred = model(x)
      y_pred = y_pred.squeeze(1)

      ## final prediction with a cut off probability
      y_prob = torch.sigmoid(y_pred)
      y_pred = (y_prob>0.5).int()

      labels.append(y.cpu())
      preds.append(y_pred.cpu())

  return torch.cat(labels, dim=0), torch.cat(preds, dim=0)


def print_report(model, test_iterator, device):
  labels, pred = predict(model, test_iterator, device)
  print(confusion_matrix(labels, pred))
  print("\n")
  print(classification_report(labels, pred))