# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:27:47 2021
@author: Safwen
"""

'''
A script for loading the data and serving it to the model for pretraining
'''
#%%
import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"),'.'))

from ctypes import cdll
cdll.LoadLibrary("libstdc++.so.6") 

from PIL import ImageFilter, ImageOps
import numpy as np
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler
from simclr_tran import TransformsSimCLR
import albumentations
from our_dataset import FlowersDataset,rotnet_collate_fn
#from utils.transformations import TransformsSimCLR
import utils
from utils.helpers import visualize
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
from PIL import Image




class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
 
def get_dataloaders(cfg,val_split=None):
    
    train_dataloaders,val_dataloaders,test_dataloaders = loaders(cfg)

    return train_dataloaders,val_dataloaders,test_dataloaders

def get_datasets(cfg):
    
    train_dataset,val_dataset,test_dataset = loaders(cfg,get_dataset=True,)

    return train_dataset,test_dataset

def loaders(cfg,get_dataset=False):
    
    if cfg.data_aug:
        print('using data aug')
        s=1

        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        transform_2 = transforms.Compose([
                #transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
                #transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, 
                expand=True, center=None, fill=0, resample=None),
                transforms.CenterCrop(80),
                transforms.Resize((96,96)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomApply([ImageOps.solarize], p=0.2),
                transforms.ToTensor(),
                
            ])
        
        data_aug = transforms.Compose([transforms.RandomResizedCrop(120, scale=(0.08, 1.)), #120
                                       
                                       transforms.Resize((192,192)), #192 
                                       transforms.ToTensor(),
                                       ])

        #data_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomCrop(32, padding=4)])
    
        if cfg.mean_norm == True:
#            transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),data_aug,
#                                        transforms.ToTensor(),
#                                        transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])
            transform = transforms.Compose([data_aug,transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])       
#        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),data_aug,
#                                        transforms.ToTensor()])                 
        #transform = transforms.Compose([data_aug,transforms.ToTensor()]) 
        transform=data_aug 
    elif cfg.mean_norm:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),
                                        transforms.ToTensor(),transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])
    else:
        print('no data aug')
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),
                                        transforms.ToTensor()])   
    
    transform_test = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),
                                        transforms.ToTensor()]) 
    
    if cfg.pretext=='rotation':
        collate_func=rotnet_collate_fn
    else:
        collate_func=default_collate
    
    

    annotation_file = 'stl.csv'                                 
    
    train_dataset = FlowersDataset(cfg,annotation_file,\
                            transform=transform)
    val_dataset=None
    
    
    annotation_file = 'val_data.csv'                                  
    
    test_dataset = FlowersDataset(cfg,annotation_file,\
                                  transform=transform_test)
    #val_dataset=test_dataset	
    # if you want to use a portion of training dataset as validation data
    if cfg.val_split:
        
#        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(cfg.val_split * dataset_size))
        # shuffle dataset
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        dataloader_train = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                                  collate_fn=collate_func,sampler=train_sampler)
        
        dataloader_val = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                       collate_fn=collate_func,sampler=valid_sampler)
    
    else:
        dataloader_train = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)
        if val_dataset:
            # if you have separate val data define val loader here
            dataloader_val = DataLoader(val_dataset,batch_size=cfg.batch_size,\
                                collate_fn=collate_func,shuffle=True)
        else:
            dataloader_val=None
    
    if get_dataset:
        
        return train_dataset,val_dataset,test_dataset
        
    dataloader_test = DataLoader(test_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)
    
    return dataloader_train,dataloader_val, dataloader_test

import yaml
if __name__=='__main__':
    class dotdict(dict):
   
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def load_yaml(config_file,config_type='dict'):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        #params = yaml.load(f,Loader=yaml.FullLoader)
        
        if config_type=='object':
            cfg = dotdict(cfg)
        return cfg
    
    config_file=r'Python Scripts/Self Supervised Learning/config/config_sl_stl.yaml'
    #config_file=r'config/config_sl.yaml'
    cfg = load_yaml(config_file,config_type='object')
    
#    tr_dset,ts_dset = get_datasets(cfg)

    tr_loaders,val_loaders,ts_loaders = get_dataloaders(cfg)
        
#    print ('length of tr_dset: {}'.format(len(tr_dset)))
#    print ('length of ts_dset: {}'.format(len(ts_dset)))

    
    data, label,idx,_ = next(iter(val_loaders))
    print(data.shape, label) 

    #data, label,idx,_ = next(iter(ts_loaders))
    #print(data.shape, label)    
    
    visualize(data.numpy(),label) 

# %%
