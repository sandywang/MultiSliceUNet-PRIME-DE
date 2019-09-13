#!/usr/bin/env python
import torch
import torch.nn as nn
from function import predict_volumes
from model import UNet2d
import os, sys
import argparse

if __name__=='__main__':
    NoneType=type(None)
    # Argument
    parser = argparse.ArgumentParser(description='Skull Stripping')
    parser.add_argument('-in', '--input_t1w', type=str, required=True, help='Input T1w Image for Skull Stripping')
    parser.add_argument('-out', '--out_dir', type=str, help='Output Dir')
    parser.add_argument('-suffix', '--mask_suffix', type=str, default="pre_mask", help='Suffix of Mask')
    parser.add_argument('-model', '--predict_model', type=str, help='Predict Model')
    parser.add_argument('-slice', '--input_slice', type=int, default=3, help='Number of Slice for Model Input')
    parser.add_argument('-conv', '--conv_block', type=int, default=5, help='Number of UNet Block')
    parser.add_argument('-kernel', '--kernel_root', type=int, default=16, help='Number of the Root of Kernel')
    parser.add_argument('-rescale', '--rescale_dim', type=int, default=256, help='Number of the Root of Kernel')
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    # Define whether show slice results
    
    train_model=UNet2d(dim_in=args.input_slice, num_conv_block=args.conv_block, kernel_root=args.kernel_root)
    checkpoint=torch.load(args.predict_model, map_location={'cuda:0':'cpu'})
    train_model.load_state_dict(checkpoint['state_dict'])
    model=nn.Sequential(train_model, nn.Softmax2d())

    predict_volumes(model, cimg_in=args.input_t1w, bmsk_in=None, rescale_dim=args.rescale_dim, save_dice=False,
            save_nii=True, nii_outdir=args.out_dir, suffix=args.mask_suffix)
