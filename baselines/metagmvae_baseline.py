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
import torchvision.transforms as transforms

import sys
sys.path.append("../")
from utils import *


def metagmvae_train(gmvae_model, opt, meta_train_set, meta_valid_set, descriptor, args):        
    print(f"[{descriptor}] Pre-Training Meta-GMVAE...")
    tb_writer = SummaryWriter(log_dir=os.path.join(LEARNCURVEDIR, descriptor, str(datetime.datetime.now())))

    global_epoch = 0
    global_step = 0
    freq_iters = 1000
    sample_size = 200
    batch_size = 4
    beta = args.gmvae_beta
    train_loader = DataLoader(meta_train_set, batch_size=batch_size*sample_size, shuffle=True, drop_last=True)

    train_iterator = iter(train_loader)

    while (global_epoch * freq_iters < args.epochs):
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
                X = X.view(batch_size, sample_size, -1, args.imgSizeToEncoder, args.imgSizeToEncoder)

                rec_loss, kl_loss = gmvae_model(X)
                loss = rec_loss + beta * kl_loss
                
                loss.backward()          
                opt.step()

                postfix = OrderedDict(
                    {'rec': '{0:.4f}'.format(rec_loss), 
                    'kld': '{0:.4f}'.format(kl_loss)
                    }
                )
                pbar.set_postfix(**postfix)                    
                pbar.update(1)
                tb_writer.add_scalar("Loss/rec_loss", rec_loss, global_step)
                tb_writer.add_scalar("Loss/kl_loss", kl_loss, global_step)
                tb_writer.add_scalar("Loss/total_loss", loss, global_step)

                global_step += 1

        global_epoch += 1

    return gmvae_model    
    
    
  
def metagmvae_test(gmvae_model, task_generator, loss_fn, descriptor, args):
    meta_test_losses, meta_test_accurs = [], []
    gmvae_model.eval()
    all_task_batch = []
    all_predictions = []
    data_transforms = transforms.Resize((
                                    args.imgSizeToEncoder, 
                                    args.imgSizeToEncoder))

    for _ in tqdm(range(NUM_TASKS_METATEST), desc='Testing tasks'):
        task_batch = task_generator.sample_task("meta_test", args)
        train_data, train_labels, _, test_data, test_labels, _ = task_batch 
        train_data, train_labels, test_data, test_labels =  \
            train_data.to(DEVICE), train_labels.to(DEVICE), test_data.to(DEVICE), test_labels.to(DEVICE)

        train_data = data_transforms(train_data.unsqueeze(0))
        train_labels = train_labels.unsqueeze(0)
        test_data = data_transforms(test_data.unsqueeze(0))
        test_labels = test_labels.unsqueeze(0)
        test_predictions = gmvae_model.prediction(train_data, train_labels, test_data)
        loss = loss_fn(test_predictions.float(), test_labels.float()).item()
        accuracy = torch.mean(torch.eq(test_predictions, test_labels).float()).item()
        meta_test_losses.append(loss)
        meta_test_accurs.append(accuracy)

        all_task_batch.append(task_batch)
        all_predictions.append(test_predictions)
    return meta_test_accurs
