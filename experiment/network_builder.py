#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:20:28 2021

@author: ben
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# local module
import settings as settings

class Net(nn.Module):
    def __init__(self, 
                 img_size, 
                 input_dim, 
                 output_dim, 
                 
                 dropout_flag, 
                 bn_flag, 
                 max_pool_flag, 
                 
                 conv_dim_list, 
                 fc_dim_list) -> None:
        super().__init__()
        
        self.conv_layers = nn.Sequential()
        
        curr_ch = input_dim
        curr_sz = img_size
        conv_stride = 1
        conv_kernel = 3

        # Build Convolutional Blocks
        for i, out_ch in enumerate(conv_dim_list):
            # pad the first layer to preserve image size
            pad = 4 if i == 0 else 0 
            
            self.conv_layers.append(nn.Conv2d(curr_ch, out_ch, conv_kernel, conv_stride, pad))
            curr_sz = (curr_sz + 2 * pad - conv_kernel) // conv_stride + 1
            
            if bn_flag:
                self.conv_layers.append(nn.BatchNorm2d(out_ch))
            self.conv_layers.append(nn.LeakyReLU())
                
            if (max_pool_flag):
                self.conv_layers.append(nn.MaxPool2d(2, 2))
                curr_sz = (curr_sz - 2) // 2 + 1
            
            if dropout_flag:
                self.conv_layers.append(nn.Dropout2d(0.2))
            curr_ch = out_ch

        # Build Fully Connected Blocks
        self.fc_layers = nn.Sequential()
        curr_fc = curr_ch * (curr_sz ** 2)
        
        for fc_out in fc_dim_list:
            self.fc_layers.append(nn.Linear(curr_fc, fc_out))
            if bn_flag:
                self.fc_layers.append(nn.BatchNorm1d(fc_out))
            self.fc_layers.append(nn.LeakyReLU())
            if dropout_flag:
                self.fc_layers.append(nn.Dropout(0.2))
            curr_fc = fc_out
            
        self.output_layer = nn.Linear(curr_fc, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.fc_layers(x)
        return self.output_layer(x) # Returns logits

def count_trainable_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def print_parameter_summary(model):
    print(f"{'Layer Name':<30} | {'Parameters':<15}")
    print("-" * 50)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            print(f"{name:<30} | {num:<15,}")
            total += num
    print("-" * 50)
    print(f"{'Total Trainable Params':<30} | {total:<15,}")

if __name__ == '__main__':
    # Example coded settings
    settings_coded_example = {
        'dataset_coded': 'CIFAR10',
        
        'batch_size_coded': -1,
        
        'dropout_flag_coded': -1,
        'bn_flag_coded': -1,
        'max_pool_flag_coded': -1,
        'initialization_coded': -1,

        'cnn_width_coded': 1,
        'conv_dim_list_coded': 1,
        'fc_width_coded': 1,
        'fc_dim_list_coded': 1
        }
    
    # convert coded settings to decoded settings
    decoded_settings = settings.decode(settings_coded_example)

    # Test network creation
    net = Net(img_size=decoded_settings['img_size_decoded'],
              output_dim=decoded_settings['output_dim_decoded'], 
              input_dim=decoded_settings['input_dim_decoded'], 

              dropout_flag=decoded_settings['dropout_flag_decoded'],
              bn_flag=decoded_settings['bn_flag_decoded'], 
              max_pool_flag=decoded_settings['max_pool_flag_decoded'],
              conv_dim_list=decoded_settings['conv_dim_list_decoded'], 
              fc_dim_list=decoded_settings['fc_dim_list_decoded']
              )
    print(net)
    print_parameter_summary(net)

    dummy_input = torch.randn(2, 
                              decoded_settings['input_dim_decoded'], 
                              decoded_settings['img_size_decoded'], 
                              decoded_settings['img_size_decoded'])
    dummy_output = net(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")  
