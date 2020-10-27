#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 01:09:33 2019

@author: swcho
"""

import os
import pickle
import scipy.io as scio
import numpy as np
import argparse

def load_features(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert2mat(base_dir, is_force=False):
    print (base_dir)
    path_pair = os.path.join(base_dir, 'sim.pkl')
    if os.path.exists(path_pair):
        data_pair   = load_features(path_pair)
    path_single = os.path.join(base_dir, 'imp.pkl')
    if os.path.exists(path_single):        
        data_single = load_features(path_single)
    path_single_vector = os.path.join(base_dir, 'imp_vector.pkl')
    if os.path.exists(path_single_vector):
        data_single_vector = load_features(path_single_vector)
    path_name = os.path.join(base_dir, 'y_name_pos.pkl')
    if os.path.exists(path_name):
        data_name = load_features(path_name)    
    
    if (not os.path.exists(os.path.join(base_dir, 'pair.mat')) and os.path.exists(path_pair)) or is_force:
        scio.savemat(os.path.join(base_dir, 'pair.mat'), mdict={'pair':data_pair})
        print ('pair.mat is generated in {}'.format(os.path.join(base_dir, 'pair.mat')))
    if (not os.path.exists(os.path.join(base_dir, 'single.mat')) and os.path.exists(path_single)) or is_force:
        scio.savemat(os.path.join(base_dir, 'single.mat'), mdict={'single':data_single})
        print ('single.mat is generated in {}'.format(os.path.join(base_dir, 'single.mat')))
    if (not os.path.exists(os.path.join(base_dir, 'single_vector.mat')) and os.path.exists(path_single_vector)) or is_force:
        scio.savemat(os.path.join(base_dir, 'single_vector.mat'), mdict={'single_vector':data_single_vector})
        print ('single_vector.mat is generated in {}'.format(os.path.join(base_dir, 'single_vector.mat')))
    if (not os.path.exists(os.path.join(base_dir, 'name.mat')) and os.path.exists(path_name)) or is_force:
        scio.savemat(os.path.join(base_dir, 'name.mat'), mdict={'name':data_name['name']})
        print ('name.mat is generated in {}'.format(os.path.join(base_dir, 'name.mat')))    

def convert2mat_npy(base_dir, is_force=False):
    path_w = os.path.join(base_dir, 'imp_cls_w.npy')
    if os.path.exists(path_w):
        data_weight = np.load(path_w)
    if (not os.path.exists(os.path.join(base_dir, 'imp_cls_w.mat')) and os.path.exists(path_w)) or is_force:
        scio.savemat(os.path.join(base_dir, 'imp_cls_w.mat'), mdict={'W_Extract':data_weight})
        print ('imp_cls_w.mat is generated in {}'.format(os.path.join(base_dir, 'imp_cls_w.mat')))
    
    path_w = os.path.join(base_dir, 'sim_cls_w.npy')
    if os.path.exists(path_w):
        data_weight = np.load(path_w)
    if (not os.path.exists(os.path.join(base_dir, 'sim_cls_w.mat')) and os.path.exists(path_w)) or is_force:
        scio.savemat(os.path.join(base_dir, 'sim_cls_w.mat'), mdict={'W_Extract':data_weight})
        print ('sim_cls_w.mat is generated in {}'.format(os.path.join(base_dir, 'sim_cls_w.mat')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl, npy to mat for DUC and TAC")
    parser.add_argument('--base_path',              default='../../data') # Capsnet_DPP
    parser.add_argument('--DUC_data_path',          default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path',          default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'])
    parser.add_argument('--is_force',               action='store_true')
    args = parser.parse_args()
    
    for data_paths in [args.DUC_data_path, args.TAC_data_path]:
        for path in data_paths:
            base_dir = os.path.join(args.base_path, path, 'BERT_features')
            convert2mat(base_dir, is_force=args.is_force)
    
    base_dir = './weights'
    convert2mat_npy(base_dir, is_force=args.is_force)
    
    
    
    