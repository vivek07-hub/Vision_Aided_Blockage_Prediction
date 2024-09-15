import os
import numpy as np
import pandas as pd
import torch
import random
from skimage import io
from torch.utils.data import Dataset
import ast

def create_samples(root, shuffle=False, nat_sort=False):
	f = pd.read_csv(root)
	data_samples = []
	beam_samples = []
	for idx, row in f.iterrows():
		bboxes = row.values[:8]
		data_samples.append(bboxes)
		beams = row.values[8:9]
		beam_samples.append(beams)

	print('list is ready')
	return data_samples, beam_samples

"""class DataFeed(Dataset):

    #A class fetching a PyTorch tensor of beam indices.

    
    def __init__(self, root_dir,
    			n,
    			transform=None,
    			init_shuflle=True):
    
        self.root = root_dir
        self.samples, self.pred_val = create_samples(self.root, shuffle=init_shuflle)
        self.transform = transform
        self.seq_len = n
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx] # Read one data sample
        beam_val = self.pred_val[idx]
        sample = sample[:self.seq_len]
        bbox_val = torch.zeros((self.seq_len,30))
        beam_val = beam_val[:1] # Read a sequence of tuples from a sample
        beams = torch.zeros((1,))
        for i,s in enumerate( beam_val ):
            x = s # Read only beams
            beams[i] = torch.tensor(x, requires_grad=False)
        	
        for i,s in enumerate(sample):
            data = s 
            bbox_data = ast.literal_eval(data)           
            bbox_val[i] = torch.tensor(bbox_data, requires_grad=False)		
        
        return (bbox_val, beams)"""


from torch.nn.utils.rnn import pad_sequence

class DataFeed(Dataset):
    def __init__(self, root_dir, n, transform=None, init_shuffle=True):
        self.root = root_dir
        self.samples, self.pred_val = create_samples(self.root, shuffle=init_shuffle)
        self.transform = transform
        self.seq_len = n
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]  # Read one data sample
        beam_val = self.pred_val[idx]
        
        # Process beam values
        beam_val = np.array(beam_val, dtype=np.float32)  # Convert to numpy array and ensure type
        beam_val = torch.from_numpy(beam_val).long()  # Convert to Torch Tensor
        
        # Initialize tensors for bbox values and the attention mask
        bbox_val = torch.zeros((self.seq_len, 30), dtype=torch.float32)
        attn_mask = torch.zeros(self.seq_len, dtype=torch.bool)  # Mask is True for positions we want to ignore
        
        # Fill in actual data and update the attention mask accordingly
        for i, s in enumerate(sample):
            data = ast.literal_eval(s) if isinstance(s, str) else s
            bbox_data = np.array(data, dtype=np.float32)  # Ensure numpy array of type float32
            bbox_val[i] = torch.from_numpy(bbox_data)
            attn_mask[i] = True if (i >= 10 or np.sum(bbox_data) == 0) else False  # Update attention mask
            if bbox_data[4]==-1:
                 #attn_mask[0:4]=False
                 #attn_mask[5:9]=False
                 for i in range(4):
                      attn_mask[i]=True
                 for i in range(5,9):
                      attn_mask[i]=True     
        
        return bbox_val, beam_val, attn_mask
    
    