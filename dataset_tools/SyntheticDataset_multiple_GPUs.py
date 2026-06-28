import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms
import random

def custom_target_transform(mask):
    mask = (np.array(mask)).astype(np.uint8) # Convert to float and normalize
    mask = np.expand_dims(mask, axis=0)  # Add channel dimensio
    mask = torch.from_numpy(mask)
    return mask

def custom_transform(image):
    # 1. Convert to numpy array
    image = np.array(image) 
    
    # 2. Convert to PyTorch tensor
    # We cast to .float() because your CNN's first convolutional layer expects floating-point weights, 
    # even if the data itself is just 0.0 and 1.0.
    image = torch.from_numpy(image).float()
    
    # 3. Ensure proper dimension order [Channels, Height, Width]
    if image.ndim == 3:
        image = image.permute(2, 0, 1)  # If RGB/Multichannel
    else:
        image = image.unsqueeze(0)      # If 2D Grayscale/Mask, add the channel dim: [1, H, W]
        
    return image

class SyntheticDataset_multiple_GPUs(Dataset):
    def __init__(self, image_dir, mask_dir, subdomains_dist=(1,3)):
        '''
        Params:
        image_dir: <string> Path to the stored images
        mask_dir: <string> Path to the corresponding stored masks
        subdomains_dist: <tuple> (height, width)
        '''

        self.img_labels = sorted([file for file in os.listdir(image_dir) if file.endswith(".png")])
        self.mask_labels = sorted([file for file in os.listdir(mask_dir) if file.endswith(".png")])
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = custom_transform
        self.target_transform = custom_target_transform
        self.subdomains_dist = subdomains_dist

    def __len__(self):
        return len(self.img_labels)
    
    def __split_image(self, full_image):
        subdomain_tensors = []
        subdomain_height = full_image.shape[1] // self.subdomains_dist[0]
        subdomain_width = full_image.shape[2] // self.subdomains_dist[1]

        for i in range(self.subdomains_dist[1]):
            for j in range(self.subdomains_dist[0]):
                subdomain = full_image[:, j * subdomain_height: (j + 1) * subdomain_height,
                            i * subdomain_width: (i + 1) * subdomain_width]
                subdomain_tensors.append(subdomain)

        return subdomain_tensors        

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        mask_name = self.mask_labels[idx]
        
        img_path =  os.path.join(self.img_dir, f"{img_name}")                
        mask_path =  os.path.join(self.mask_dir, f"{mask_name}")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert('L') 

        images = []

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            
        images = self.__split_image(image)      

        return images, mask
