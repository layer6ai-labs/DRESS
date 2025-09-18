import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../')
from utils import *


def contrastive_pretrain(model, optimizer, train_set, descriptor, args):
    print(f"[{descriptor}] Pre-Training the encoder...")
    train_loader =  DataLoader(train_set, 
                               batch_size=PRETRAIN_BATCH_SIZE, 
                               shuffle=True, 
                               drop_last=False)
    data_transforms_for_encoder = transforms.Resize((
                                    args.imgSizeToEncoder, 
                                    args.imgSizeToEncoder))
    criterion = nn.CrossEntropyLoss()
    for epoch in trange(PRETRAIN_EPOCHS):
        model.train()
        train_loss = []
        for i, (images, _) in enumerate(tqdm(train_loader)): 
            x1 = data_transforms_for_encoder(images[0]).to(DEVICE, non_blocking=True, dtype=torch.float)
            x2 = data_transforms_for_encoder(images[1]).to(DEVICE, non_blocking=True, dtype=torch.float)
            # compute output and loss
            z1, z2 = model(x1, x2)
            logits, labels = model.info_nce_loss(z1, z2, DEVICE, temperature= 0.5)
            loss = criterion(logits, labels)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
        train_loss = np.mean(train_loss)
        print(f'{descriptor} Training epochs: {epoch}, loss: {train_loss:.4f}')                 
  
    print(f"[{descriptor}] Training function completed!")
    return model


def test_pretrain(model, task_generator, descriptor, args):
    data_transforms_for_encoder = transforms.Resize((
                                    args.imgSizeToEncoder, 
                                    args.imgSizeToEncoder))
    task_batch = task_generator.sample_task("meta_test", args)
    test_losses, test_accures = [], [] 
    for task_id in tqdm(range(NUM_TASKS_METATEST), desc='Testing tasks'):
        task_batch = task_generator.sample_task("meta_test", args)
        K = args.KShot
        train_data, train_labels, _, test_data, test_labels, _ = task_batch
        train_data, test_data = data_transforms_for_encoder(train_data), \
                                    data_transforms_for_encoder(test_data)
        train_data, train_labels = train_data.to(DEVICE), train_labels.to(DEVICE)
        test_data, test_labels = test_data.to(DEVICE), test_labels.to(DEVICE)
        with torch.no_grad():
            train_reps = model.get_representations(train_data)
            test_reps = model.get_representations(test_data)
            
        # train a linear model
        linear_model = nn.Linear(model.feature_dim, args.NWay).to(DEVICE)
        linear_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(linear_model.parameters(), lr=FINETUNE_LR, weight_decay=0)
        for step in range(FINETUNE_STEPS):
            train_loss = criterion(linear_model(train_reps), train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                linear_model.eval()
                test_preds = linear_model(test_reps)
                test_loss, test_accur = criterion(test_preds, test_labels), accuracy_fn(test_preds, test_labels)
        test_losses.append(test_loss.item())
        test_accures.append(test_accur.item())
    
    print(f"[{descriptor}] testing completed!")
    return test_accures

