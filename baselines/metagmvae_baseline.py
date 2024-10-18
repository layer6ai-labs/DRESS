"""
Meta-GMVAE Code Adapted from https://github.com/db-Lee/Meta-GMVAE
"""

import os
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import sys
sys.path.append("../")
from utils import *


def metagmvae_train(gmvae_model, opt, meta_train_set, meta_valid_set, descriptor, args):        
    print(f"[{descriptor}] Pre-Training Meta-GMVAE...")

    global_epoch = 0
    freq_iters = 1000
    sample_size = 200
    batch_size = 4

    train_loader =  DataLoader(meta_train_set, batch_size=batch_size*sample_size, shuffle=True, drop_last=False)

    train_iterator = iter(train_loader)

    while (global_epoch * freq_iters < METATRAIN_OUTER_EPISODES):
        with tqdm(total=freq_iters) as pbar:
            for _ in range(freq_iters):
                gmvae_model.train()
                gmvae_model.zero_grad()

                try:
                    X = next(train_iterator)[0]
                except StopIteration:
                    iterator = iter(train_loader)
                    X = next(iterator)[0]
                                    
                X = X.to(DEVICE).float()
                X = X.view(batch_size, sample_size, args.imgSizeToEncoder, args.imgSizeToEncoder)

                rec_loss, kl_loss = gmvae_model(X)
                loss = rec_loss + kl_loss
                
                loss.backward()          
                opt.step()

                postfix = OrderedDict(
                    {'rec': '{0:.4f}'.format(rec_loss), 
                    'kld': '{0:.4f}'.format(kl_loss)
                    }
                )
                pbar.set_postfix(**postfix)                    
                pbar.update(1)

        global_epoch += 1

    return gmvae_model    
    
    
  
def metagmvae_test(gmvae_model, task_generator, loss_fn, descriptor, args):
    meta_test_losses, meta_test_accurs = [], []
    gmvae_model.eval()

    for _ in tqdm(range(NUM_TASKS_METATEST), desc='Testing tasks'):
        task_batch = task_generator.sample_task("meta_test", args)
        train_data, train_labels, _, test_data, test_labels, _ = task_batch 
        train_data, train_labels, test_data, test_labels =  \
            train_data.to(DEVICE), train_labels.to(DEVICE), test_data.to(DEVICE), test_labels.to(DEVICE)

        test_predictions = gmvae_model.prediction(train_data, train_labels, test_data)
        loss = loss_fn(test_predictions, test_labels).item()
        accuracy = torch.mean(torch.eq(test_predictions, test_labels).float()).item()
        meta_test_losses.append(loss)
        meta_test_accurs.append(accuracy)
    
    with open("res.txt", "a") as f:
        f.write(str(datetime.datetime.now())+f' under seed {args.seed}'+'\n')
        f.write(f"[{descriptor} {args.NWay}-way {args.KShotMetaTr}-shot meteTrain {args.KShotMetaVa}-shot metaTest]: " + \
                f"Meta test loss: Mean: {np.mean(meta_test_losses):.2f}; Std: {np.std(meta_test_losses):.2f}\n" + \
                f"Meta test accuracy: Mean: {np.mean(meta_test_accurs)*100:.2f}%; Std: {np.std(meta_test_accurs)*100:.2f}%\n")
    print(f"[{descriptor}] testing completed!")
    return





