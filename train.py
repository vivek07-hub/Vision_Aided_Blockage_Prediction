import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as trf
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import random
import math



from data_feed import DataFeed  # Assuming DataFeed is a custom module


options_dict = {
    'tag': 'Exp1_beam_seq_pred_no_images',
    'operation_mode': 'beams',

    # Data:
    'trn_data_file': '/home/ai23mtech14008/Vision_Aided_Blockage_prediction/scenario17/future-1/scenario17_dev_series_train_final.csv',
    'val_data_file': '/home/ai23mtech14008/Vision_Aided_Blockage_prediction/scenario17/future-1/scenario17_dev_series_val_final.csv',



    # Train param
    'gpu_idx': 0,
    'solver': 'Adam',
    'shf_per_epoch': True,
    'num_epochs': 100,
    'batch_size': 32,
    'val_batch_size': 32,
    'lr': 1e-3,
    'lr_sch': [50, 80],
    'lr_drop_factor': 0.1,
    'wd': 0,
    'display_freq': 10,
    'coll_cycle': 10,
    'val_freq': 20,
    'prog_plot': False,
    'fig_c': 0,
    'val': False,
    'resume_train': False
}




# Data preprocessing transformations
transf = trf.Compose([
    trf.ToTensor(),
])

trn_feed = DataFeed(root_dir=options_dict['trn_data_file'],
                    n=30,
                    transform=transf)

trn_loader = DataLoader(trn_feed, batch_size=options_dict['batch_size'])
options_dict['train_size'] = trn_feed.__len__()

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU and CUDA devices
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # The following two settings can help with reproducibility,
    # but may affect performance and are not necessary in all cases.
    # They ensure that the same convolution algorithms are selected every time,
    # potentially at the cost of reduced performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Example seed




# Define a simple Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes, dim_feedforward=512, num_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_feedforward, num_classes)
    
    def forward(self, src, src_key_padding_mask):
        src = self.embedding(src)  # embedding the input
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # Pass through the transformer with mask
        output = torch.mean(output, dim=1)  # average pooling
        #output = self.dropout(output)
        output = self.output_layer(output)  # final output layer
        return output

# Hyperparameters
input_dim = 30
num_heads = 2
num_classes = 2 # Adjust this to your number of classes
dim_feedforward = 16
num_layers = 2
#batch_size = 16
learning_rate = 0.001
num_epochs = 75

# Model, loss, and optimizer
model = TransformerModel(input_dim, num_heads, num_classes, dim_feedforward, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)



# Calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for bbox_val, beams, attn_mask in data_loader:  # Updated to include attn_mask
            outputs = model(bbox_val, src_key_padding_mask=attn_mask)  # Use the attention mask
            _, predicted = torch.max(outputs, 1)
            total += beams.size(0)
            correct += (predicted == beams.squeeze()).sum().item()
    accuracy = correct / total
    return accuracy
   

# Assume a function to calculate accuracy as shown in the previous response
"""accuracy = calculate_accuracy(model, trn_loader)
print(f'Accuracy: {accuracy:.4f}')"""


# Prepare the validation data loader
val_feed = DataFeed(root_dir=options_dict['val_data_file'], n=30, transform=transf)
val_loader = DataLoader(val_feed, batch_size=options_dict['val_batch_size'])

# Function to perform a training epoch
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for bbox_val, beams, attn_mask in data_loader:
        optimizer.zero_grad()
        outputs = model(bbox_val, src_key_padding_mask=attn_mask)
        loss = criterion(outputs, beams.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Function to evaluate the model
def evaluate(model, data_loader):
    return calculate_accuracy(model, data_loader)

# Training loop with both training and validation
for epoch in range(num_epochs):
    train_loss = train_epoch(model, trn_loader, optimizer, criterion)
    val_accuracy = evaluate(model, val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


# Step 1: Prepare the Test DataLoader
# Define the path to your test dataset
test_data_file = 'Vision_Aided_Blockage_prediction/scenario17/future-1/scenario17_dev_series_val_final.csv'  # Update this path

# Load your test data
test_feed = DataFeed(root_dir=test_data_file, n=30, transform=transf)
test_loader = DataLoader(test_feed, batch_size=options_dict['val_batch_size'])  # Assuming you use the same batch size as validation

def calculate_test_accuracy(model, test_loader):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient calculation for efficiency
        for bbox_val, beams, attn_mask in test_loader:
            outputs = model(bbox_val, src_key_padding_mask=attn_mask)
            _, predicted = torch.max(outputs, 1)
            total += beams.size(0)
            correct += (predicted == beams.squeeze()).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Now call this function with the test_loader
calculate_test_accuracy(model, test_loader)


