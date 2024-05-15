##########################################################################################
# Importing libraries

import os, random, time, multiprocessing, glob, cv2, numpy as np, pandas as pd, nibabel as nib, matplotlib.pylab as plt

# Pytorch functions
import torch
# Neural network layers
import torch.nn as nn
import torch.nn.functional as F
# Optimizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
# Torchvision library
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# For results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from torchsummary import summary

from utilities import *
from preprocessing_utilities import *

##########################################################################################
# User defined parameters

SEED = 44
USE_SEED = True
N_EPOCHS_CAE = 40
N_EPOCHS_UNet = 50
batch_size = 64
scan_type = 'Flair'

master_path = r'./BraTS/'
folders = [folder for folder in os.listdir(os.path.join(master_path, 'BraTS2021_Training_Data')) if folder != '.DS_Store']

##########################################################################################
# Split the dataset

dataset_indices = list(range(len(folders)))
train_indices, test_indices = train_test_split(dataset_indices, test_size=0.1, random_state=SEED)
train_indices, val_indices = train_test_split(train_indices, test_size=0.22, random_state=SEED)

train_folders = [folders[i] for i in train_indices]
valid_folders = [folders[i] for i in val_indices]
test_folders = [folders[i] for i in test_indices]

##########################################################################################
# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
   device = torch.device('mps')
else:
    device = torch.device('cpu')

def set_seed(seed, use_cuda = True, use_mps = False):
    """
    Set SEED for PyTorch reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if use_mps:
        torch.mps.manual_seed(seed)


if USE_SEED:
    set_seed(SEED, torch.cuda.is_available(), torch.backends.mps.is_available())

##########################################################################################
## Define Custom Dataset
class BraTSDataset(Dataset):
    def __init__(self, image_path = r'./BraTS/BraTS2021_Training_Data_Slice', transform=None):
        'Initialisation'
        self.image_path = image_path
        self.folders_name = [folder for folder in os.listdir(self.image_path) if folder != '.DS_Store']
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.folders_name) * 155

    def __getitem__(self, index):
        'Generates one sample of data'

        # Determine the image index and the RGB layer
        image_idx = index // 155
        layer_idx = index % 155

        # Select sample
        file_name = self.folders_name[image_idx]
        
        path_img = os.path.join(self.image_path, file_name, scan_type.lower(), file_name + '_' + scan_type.lower() + '_' + str(layer_idx+1) + '.npy')
        image = np.load(path_img).astype(np.float32)

        path_label = os.path.join(self.image_path, file_name, 'seg', file_name + '_seg_' + str(layer_idx+1) + '.npy')
        label = np.load(path_label)
        
        if self.transform:
            image, label = self.transform([image, label])
        return image, label
    
class BinariseLabel(object):
    def __call__(self, sample):
        image, label = sample
        new_label = np.sign(label)
        return image, new_label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample

        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(label)
    
dataset = BraTSDataset(image_path = r'./BraTS/BraTS2021_Training_Data_Slice',
                        transform=transforms.Compose([
                            BinariseLabel(),
                            ToTensor()
                        ]))

##########################################################################################
# Train Test Split

tmp_list = [[],[],[]]
for i, ind_list in enumerate([train_indices, val_indices, test_indices]):
    for ind in ind_list:
        for j in range(155):
            tmp_list[i].append(ind*155 + j)
train_indices, val_indices, test_indices = tmp_list

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)
test_subset = Subset(dataset, test_indices)

# Create the subset DataLoader
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

##########################################################################################
## Define Convlutional Autoencoder Structure
class ConvAutoencoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.features = nn.Sequential(
      ## encoder layers ##
      # conv layer (depth from 1 --> 4), 3x3 kernels
      # Input 64 x 64
      nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding = 'same'), # 64 x 64
      nn.ReLU(),
      # pooling layer to reduce x-y dims by two; kernel and stride of 2
      nn.MaxPool2d(2), ## 32 x 32
      # conv layer (depth from 4 --> 8), 4x4 kernels
      nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding = 'same'), # 32 x 32
      nn.ReLU(),
      nn.MaxPool2d(2), # 16 x 16
      # conv layer (depth from 8 --> 12), 5x5 kernels
      nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding = 'same'), # ( 12 x ) 16 x 16
      nn.ReLU(),
      
      ## decoder layers ##
      # add transpose conv layers, with relu activation function
      nn.ConvTranspose2d(12, 6, kernel_size = 2, stride=2), # 32 x 32
      nn.ReLU(),
      nn.ConvTranspose2d(6, 1, kernel_size = 2, stride=2), # 64 x 64
      # output layer (with sigmoid for scaling from 0 to 1)
      # nn.Sigmoid()
    )
    
  def forward(self, x):
    x = x.view(int(np.prod(x.shape)/(64**2)), 1, 64, 64)
    x = self.features(x)
    # x = x.view(x.shape[0], -1)
    # x = torch.flatten(x, start_dim=1)
    return x

model = ConvAutoencoder().to(device)

##########################################################################################
# Loss function and Optimizer

# Loss function
criterion = torch.nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# Optim
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

##########################################################################################
# Load trained model

model.load_state_dict(torch.load(f'./models/CA_{scan_type}.pt'))
model.to(device)

##########################################################################################
# Test prediction model

labels, preds = predict(model, test_dataloader, device)

##########################################################################################
# Preprocessing for the UNet model prediction

def index_converter(index):
    return index // 155, 1 + index % 155 # image_idx, layer_idx

os.makedirs(os.path.join('./BraTS', f'CA_{scan_type}_Area'), exist_ok=True)

pred_no_tumour = []
pred_tumour = []
for i in range(len(test_indices)):
    image_idx, layer_idx = index_converter(test_indices[i])
    image_idx = folders[image_idx]
    pred_label = preds[i]

    rows_indices = torch.where(pred_label)[0]
    cols_indices = torch.where(pred_label)[1]
    if len(rows_indices) == 0 or len(cols_indices) == 0:
        pred_no_tumour.append([str(image_idx).zfill(5), str(layer_idx)])
        continue
    pred_tumour.append([str(image_idx).zfill(5), str(layer_idx)])
    top_row = torch.min(rows_indices)
    bottom_row = torch.max(rows_indices)
    left_col = torch.min(cols_indices)
    right_col = torch.max(cols_indices)

    width = right_col - left_col + 1
    height = bottom_row - top_row + 1

    if width > height:
        top_row = top_row - np.floor((width - height) / 2)
        bottom_row = bottom_row + np.ceil((width - height) / 2)
        if top_row < 0:
            bottom_row = bottom_row - top_row
            top_row = 0
        elif bottom_row > 63:
            top_row = top_row - (bottom_row - 63)
            bottom_row = 63
    elif height > width:
        left_col = left_col - np.floor((height - width) / 2)
        right_col = right_col + np.ceil((height - width) / 2)
        if left_col < 0:
            right_col = right_col - left_col
            left_col = 0
        elif right_col > 63:
            left_col = left_col - (right_col - 63)
            right_col = 63

    path = os.path.join('./BraTS', f'CA_{scan_type}_Area', str(image_idx).zfill(5) + '_ROI_pred_' + str(layer_idx))
    np.save(path, np.array([top_row, bottom_row, left_col, right_col]).astype(int))

##########################################################################################
# Preprocessing from 'CA_{scan_type}_Area' to 'UNet_Test_Input'

os.makedirs(os.path.join(master_path, 'UNet_Test_Input'), exist_ok=True)

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use all available CPU cores
pool.map(convert_Unet_pred, pred_tumour)
pool.close()
pool.join()

##########################################################################################
# Load the UNet model

testing_files = np.sort([image for image in os.listdir('./BraTS/UNet_Test_Input') if image != '.DS_Store'])

class BraTSDataset(Dataset):
    def __init__(self, image_path = './BraTS/UNet_Test_Input', transform = None):
        'Initialisation'
        self.image_names = np.sort([image for image in os.listdir('./BraTS/UNet_Test_Input') if image != '.DS_Store'])
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image_name = self.image_names[index]
        
        path_img = os.path.join(self.image_path, image_name)
        image = np.load(path_img).astype(np.float32)
        label = image
        
        if self.transform:
            image, label = self.transform([image, label])
        return torch.from_numpy(image), torch.from_numpy(label)

dataset = BraTSDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

##########################################################################################
# The UNet model architecture

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        ## Input is 32 x 32 x 1
        ## Output is 32 x 32 x 4
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3 32 x 32 x 1
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 30x30x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 64, 64)
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out

model = UNet().to(device)

##########################################################################################
# Loss function and Optimizer

# Loss
criterion = torch.nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# Optim
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

##########################################################################################
# Load trained model

model.load_state_dict(torch.load(f'./models/Unet_{scan_type}.pt'))
model.to(device)

##########################################################################################
# Prediction

_, preds_UNet = predict(model, dataloader, device)

##########################################################################################
# DICE Accuracy for the full model

def extract_id(name):
    name = os.path.splitext(name)[0]
    return name.split('_')[1], name.split('_')[-1]

def rebuild_prediction(image):
    image = torch.argmax(image, dim=0)
    image = torch.where(image == 3, torch.tensor(4), image)
    return image

def dice2d(pred, target):
    return (pred == target).sum() / np.prod(pred.shape)

def dice2d_exclude_zero(pred, target):
    if np.prod(pred.shape) == np.sum((pred == 0) & (target == 0)):
        return 1
    return ((pred == target) & ((pred != 0) | (target != 0))).sum() / (np.prod(pred.shape)-np.sum((pred == 0) & (target == 0)))

list_full_test_images = [str(folders[index_converter(test_indices[i])[0]])+'_seg_'+str(index_converter(test_indices[i])[1])+'.npy' for i in range(len(test_indices))]
list_actual_input = [image for image in os.listdir('./BraTS/BraTS2021_Training_Data_2D_Unet/test/seg') if image != '.DS_Store']
list_actual_non_input = list(set(list_full_test_images).difference(set(list_actual_input)))
list_predict_input = [pred_tumour[i][0] + '_seg_' + pred_tumour[i][1] + '.npy' for i in range(len(pred_tumour))]
list_predict_non_input = [pred_no_tumour[i][0] + '_seg_' + pred_no_tumour[i][1] + '.npy' for i in range(len(pred_no_tumour))]

dices = []
dices_exclude_zero = []

# loop through each image of preds_UNet
for i in range(len(preds_UNet)):
    
    scan_name = 'BraTS2021_' + extract_id(testing_files[i])[0]
    scan_no = extract_id(testing_files[i])[1]
    dim_name = scan_name + '_ROI_pred_' + scan_no + '.npy'
    top_row, bottom_row, left_col, right_col = np.load(os.path.join('./BraTS/CA_Flair_Area', dim_name)).astype(np.int32)
    org_size = bottom_row - top_row + 1

    pred = preds_UNet[i]
    pred = rebuild_prediction(pred)
    pred_resize = cv2.resize(pred.numpy(), [org_size, org_size], interpolation=cv2.INTER_NEAREST)

    pred = np.zeros((64, 64))
    if right_col < left_col:
        left_col, right_col = right_col, left_col
    if bottom_row < top_row:
        top_row, bottom_row = bottom_row, top_row
    
    if bottom_row - top_row == right_col - left_col:
        pred[top_row:bottom_row+1, left_col:right_col+1] = pred_resize
    elif bottom_row - top_row > right_col - left_col:
        pred[top_row:bottom_row, left_col:right_col+1] = pred_resize
    else:
        pred[top_row:bottom_row+1, left_col:right_col] = pred_resize
    
    seg_name = scan_name + '_seg_' + scan_no + '.npy'
    seg = np.load(os.path.join('./BraTS','BraTS2021_Training_Data_2D', scan_name, 'seg', seg_name))
    
    dices.append(dice2d(pred, seg))
    dices_exclude_zero.append(dice2d_exclude_zero(pred, seg))

# Loop through list of not predicted but should have
should_have_predicted = list(set(list_actual_input).difference(set(list_predict_input)))

# Read each above and do np.count_nonzero(tmp == 0)/np.prod(tmp.shape)
for image in should_have_predicted:
    img_path = os.path.join('./BraTS', 'BraTS2021_Training_Data_2D', 'BraTS2021_' + image.split('_')[1], 'seg', image)
    img = np.load(img_path)
    dice = np.count_nonzero(img == 0)/np.prod(img.shape)
    dices.append(dice)

len_test = len(list_full_test_images)
len_predicted = len(list_predict_input)
len_should_have_predicted = len(should_have_predicted)
len_did_not_pred = len(set(list_predict_non_input).intersection(set(list_actual_non_input)))

print('Analysis of the final model:\nThe overall Dice Score is:', np.mean(dices) * (len_should_have_predicted + len_predicted)/len_test + len_did_not_pred/len_test, '\n\nOut of', len_test, 'test images.', len_predicted, 'images were predicted to contain a tumour, which includes', len(set(list_predict_input).difference(set(list_actual_input))), 'images which do not contain a tumour in the ground truth. Furthermore,', len(set(list_predict_non_input).intersection(set(list_actual_non_input))), 'images were correctly predicted not to contain a tumour. Finally,', len_should_have_predicted, 'images containing tumours were not detected. Therefore, the false negative rate for detecting for a tumour is', len_should_have_predicted / len_test, '.')