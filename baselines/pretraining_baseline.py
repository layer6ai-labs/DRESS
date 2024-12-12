import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import datetime
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../')
from utils import *


def accuracy_fn(preds, labels):
    preds = preds.argmax(dim=1).view(labels.shape)
    return (preds==labels).sum().float() / labels.size(0)

def contrastive_pretrain(model, optimizer, train_set, valid_set, descriptor, args):
    print(f"[{descriptor}] Pre-Training the encoder...")
    train_loader =  DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    tb_writer = SummaryWriter(log_dir=os.path.join(LEARNCURVEDIR, descriptor))
    step = 0
    # for i in tqdm(range(args.train_epochs), desc='Training Epochs'):
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        for i, (images, _) in enumerate(tqdm(train_loader, desc='Training Batches')): 
            x1 = images[0].to(DEVICE, non_blocking=True, dtype=torch.float)
            x2 = images[1].to(DEVICE, non_blocking=True, dtype=torch.float)
            # compute output and loss
            z1, z2 = model(x1, x2)
            logits, labels = model.info_nce_loss(z1, z2, DEVICE, temperature= 0.5)
            loss = criterion(logits, labels)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
            tb_writer.add_scalar("Loss/train_by_step", loss.detach().item(), step)
            step += 1
        train_loss = np.mean(train_loss)
        tb_writer.add_scalar("Loss/train", train_loss, i)
        torch.save(model.state_dict(), os.path.join(MODELDIR, f"{descriptor}_{epoch}.pth"))
        now = datetime.datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")} Training epochs: {epoch}, loss: {train_loss:.4f}')            
        if (epoch+1) % args.eval_interval == 0 or epoch == args.train_epochs - 1:
            train_acc, valid_acc = linear_eval(model, descriptor, train_loader, valid_loader, args)
            print(f"Linear eval of epoch {epoch}: Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}")
            now = datetime.datetime.now()
            print(f'{now.strftime("%Y-%m-%d %H:%M:%S")} Training epochs: {epoch}, \
                    Linear eval acc: {train_acc:.4f}, Linear eval acc: {valid_acc:.4f}')      
            tb_writer.add_scalar("Linear_eval_acc/train", train_acc, epoch)
            tb_writer.add_scalar("Linear_eval_acc/valid", valid_acc, epoch)
  
    # within the call of close(), flush() should also be called upon the tensorboard writer
    tb_writer.close()
    print(f"[{descriptor}] Training function completed!")
    return model

def linear_eval(model, descriptor, train_loader, valid_loader, args, attributes_idx=0):
    tb_writer = SummaryWriter(log_dir=os.path.join(LEARNCURVEDIR, descriptor))
    model.eval()
    num_classes = len(next(iter(train_loader))[-1][:, attributes_idx].unique())
    linear_model = nn.Linear(model.feature_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_model.parameters(), lr=args.eval_lr, weight_decay=0)
    #TODO: lr scheduler
    step = 0
    for epoch in tqdm(range(args.eval_epochs), desc='Training Epochs for linear evaluation'):
        linear_model.train()
        train_loss, valid_loss = [], []
        train_correct, valid_correct = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            if i > 2: break #mini-batch
            x = images[0].to(DEVICE, non_blocking=True, dtype=torch.float)
            y = torch.tensor(labels[:, attributes_idx]).to(DEVICE, non_blocking=True, dtype=torch.long)
            z = model.get_representations(x)
            logits = linear_model(z)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
            correct = (logits.argmax(dim=1) == y).sum().item()
            train_correct += correct
            tb_writer.add_scalar(f"Linear_eval_loss/attribute_{attributes_idx}/train", loss.detach().item(), step)
            tb_writer.add_scalar(f"Linear_eval_acc/attribute_{attributes_idx}/train", correct/len(labels), step)
            step += 1
        train_loss = np.mean(train_loss)
        train_acc = train_correct / len(train_loader.dataset)
        linear_model.eval()
        with torch.no_grad():
            for j, (images, labels) in enumerate(valid_loader):
                x = images.to(DEVICE, non_blocking=True, dtype=torch.float)
                y = labels[:, attributes_idx].to(DEVICE, non_blocking=True, dtype=torch.long)
                z = model.get_representations(x)
                logits = linear_model(z)
                loss = criterion(logits, y)
                valid_loss.append(loss.detach().item())
                correct = (logits.argmax(dim=1) == y).sum().item()
                valid_correct += correct
            valid_loss = np.mean(valid_loss)
            valid_acc = valid_correct / len(valid_loader.dataset)
            tb_writer.add_scalar(f"Linear_eval_loss/attribute_{attributes_idx}/valid", valid_loss, epoch)
            tb_writer.add_scalar(f"Linear_eval_acc/attribute_{attributes_idx}/valid", valid_acc, epoch)
        print(f"Linaer evaluation Epoch {epoch}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, \
                Valid loss: {valid_loss:.4f}, Valid acc: {valid_acc:.4f}")
    tb_writer.close()
    return train_acc, valid_acc


def test_pretrain(model, task_generator, descriptor, args):
    task_batch = task_generator.sample_task("meta_test", args)
    test_losses, test_accures = [], [] 
    test_accures_logistic = []
    for task_id in tqdm(range(NUM_TASKS_METATEST), desc='Testing tasks'):
        task_batch = task_generator.sample_task("meta_test", args)
        K = args.KShot
        train_data, train_labels, _, test_data, test_labels, _ = task_batch
        train_data = train_data.to(DEVICE)
        test_data = test_data.to(DEVICE)
        with torch.no_grad():
            train_reps = model.get_representations(train_data)
            test_reps = model.get_representations(test_data)

            # logistic regression
            logistic_model = LogisticRegression(max_iter=5000)
            train_reps_lr = train_reps.detach().cpu().numpy()
            test_reps_lr = test_reps.detach().cpu().numpy()
            logistic_model.fit(train_reps_lr, train_labels.detach().numpy())
            test_preds_lr = logistic_model.predict(test_reps_lr)
            test_acc_lr = (test_preds_lr == test_labels.detach().numpy()).mean()
            test_accures_logistic.append(test_acc_lr)
    
        train_labels = train_labels.to(DEVICE)
        test_labels = test_labels.to(DEVICE)
            
        # train a linear model
        linear_model = nn.Linear(model.feature_dim, args.NWay).to(DEVICE)
        linear_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(linear_model.parameters(), lr=args.meta_test_eval_lr, weight_decay=0)
        tb_writer = SummaryWriter(log_dir=os.path.join(LEARNCURVEDIR, descriptor))
        for step in range(args.meta_test_eval_steps):
            train_loss = criterion(linear_model(train_reps), train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            tb_writer.add_scalar("meta-test-loss/train_{task_id}", train_loss.item(), step)
            with torch.no_grad():
                linear_model.eval()
                test_preds = linear_model(test_reps)
                test_loss, test_accur = criterion(test_preds, test_labels), accuracy_fn(test_preds, test_labels)
            tb_writer.add_scalar("meta-test-acc/test_{task_id}", train_loss.item(), step)
            tb_writer.add_scalar("meta-test-acc/test_{task_id}", test_accur.item(), step)
        test_losses.append(test_loss.item())
        test_accures.append(test_accur.item())
    if not os.path.exists(RESULTSDIR):
        os.makedirs(RESULTSDIR)
    res_path = f'{args.dsName}_{args.method}_{args.backbone}'
    with open(os.path.join(RESULTSDIR, f"{res_path}_results.txt"), 'a') as f:
        f.write(str(datetime.datetime.now())+f' under seed {args.seed}'+'\n')
        f.write(f"[{res_path} {args.NWay}-{args.KShot}-shot metaTest] with linear layer: " + \
                f"Test loss: Mean: {np.mean(test_losses):.2f}; Std: {np.std(test_losses):.2f}\n" + \
                f"Test accuracy: Mean: {np.mean(test_accures)*100:.2f}%; Std: {np.std(test_accures)*100:.2f}%\n")
        f.write(f"[{descriptor} {args.NWay}-{args.KShot}-shot metaTest] with logistic regression:" + \
                f"Test accuracy: Mean: {np.mean(test_accures_logistic)*100:.2f}%; Std: {np.std(test_accures_logistic)*100:.2f}%\n")
    print(f"[{descriptor}] testing completed!")
    return test_losses, test_accures, test_accures_logistic

