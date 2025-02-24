import sys
sys.path.append("../../")
sys.path.append("../")

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

# PsCo imports
from psco.putils import *
import psco.models as models
from psco.augmentation import get_augmentation, MultipleTransform
from psco.lightning import PsCoLightning

# DRESS imports
from utils import get_args_parser, fix_seed, NUM_TASKS_METATEST, NUM_TASKS_METAVALID
from dataset_loaders import LOAD_DATASET
from task_generator import TaskGenerator

from exp_tracker import ExperimentTracker
from data_pipeline import TasksDataset

cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    # DRESS Parser
    parser = get_args_parser()
    
    # Append PsCo asrguments    
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--aug', type=str, default=['strong', 'weak'], nargs='+')

    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--eval-fewshot-metric', type=str, default='supcon')

    parser.add_argument('--model', type=str, default='psco')
    parser.add_argument('--backbone', type=str, default='conv5')

    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--queue-size', type=int, default=16384)
    parser.add_argument('--num-shots', type=int, default=4)
    parser.add_argument('--shot-sampling', type=str, default='topk', choices=['topk', 'prob'])
    parser.add_argument('--temperature2', type=float, default=1.)
    parser.add_argument('--sinkhorn-iter', type=int, default=3)

    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    parser.add_argument('--num-tasks', type=int, default=600)

    parser.add_argument('--master-port', type=int, default=2222)
    parser.add_argument('--device-id', type=int, default=0)

    args = parser.parse_args()
    
    args.lr = args.base_lr * args.batch_size / 256
    
    fix_seed(args.seed)
    
    (
        meta_train_set, 
        meta_valid_set, 
        meta_test_set, 
        meta_train_partitions_supervised, 
        meta_train_partitions_supervised_all,
        meta_train_partitions_supervised_oracle,
        meta_valid_partitions, 
        meta_test_partitions
    ) = LOAD_DATASET[args.dsName](args)
    
    transforms = dict(
        train=MultipleTransform([get_augmentation(args.dsName, aug) for aug in ["strong", "strong"]]),
        valid=get_augmentation(args.dsName, "none"),
        test=get_augmentation(args.dsName, "none"))
    
    meta_train_set.transform = transforms["train"]
    meta_valid_set.transform = transforms["valid"]
    meta_test_set.transform = transforms["test"]
    
    task_generator = TaskGenerator(
        meta_train_set, 
        meta_valid_set, 
        meta_test_set,
        meta_train_partitions_supervised,
        meta_valid_partitions,
        meta_test_partitions,
        args)
    
    ds_val_tasks = TasksDataset(
        args=args, 
        num_tasks=NUM_TASKS_METAVALID, 
        meta_split="meta_valid", 
        task_generator=task_generator)
    
    ds_test_tasks = TasksDataset(
        args=args, 
        num_tasks=NUM_TASKS_METATEST, 
        meta_split="meta_test", 
        task_generator=task_generator)

    dl_train = DataLoader(meta_train_set, batch_size=args.batch_size, shuffle=True, 
                          num_workers=0, drop_last=True)
    dl_valid = DataLoader(ds_val_tasks, batch_size=1, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test_tasks, batch_size=1, shuffle=False, num_workers=0)

    psco_model = PsCoLightning(
        model_name=args.model,
        backbone=args.backbone,
        input_shape=meta_train_set[0][0][0].shape, 
        num_train_batches=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        momentum=args.momentum,
        num_epochs=args.num_epochs,
        prediction=args.prediction,
        queue_size=args.queue_size,
        shot_sampling=args.shot_sampling,
        sinkhorn_iter=args.sinkhorn_iter,
        temperature=args.temperature,
        temperature2=args.temperature2,
        K=args.KShot,
        N=args.NWay,
        Q=args.KQuery,
        metric="supcon", 
        random_seed=args.seed)
    
    exp_tracker = ExperimentTracker(
        results_dir="output",
        config_path=None,
        name=f"psco_{args.dsName}",
        version_name=None)
    
    tb_logger = TensorBoardLogger(
        save_dir=exp_tracker.results_dir, 
        name=exp_tracker.name, 
        version=exp_tracker.version)
    
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_tracker.path, "checkpoints"),
        auto_insert_metric_name=False,
        filename='epoch={epoch}-step={step}-val_accuracy={val/accuracy:.6f}',
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_last=True,
        verbose=True)
    
    early_stopping = EarlyStopping(
        monitor='val/accuracy',
        patience=50,
        verbose=True,
        mode='max')
    
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices=[args.device_id],
        num_sanity_val_steps=2, 
        val_check_interval=0.1,
        log_every_n_steps=100, 
        logger=tb_logger,
        callbacks=[model_checkpoint, early_stopping])
    
    trainer.fit(psco_model, train_dataloaders=dl_train, val_dataloaders=dl_valid)
    
    best_model_path = model_checkpoint.best_model_path    
    best_model = PsCoLightning.load_from_checkpoint(
        best_model_path, 
        args=args, 
        input_shape=meta_train_set[0][0][0].shape, 
        num_train_batches=args.batch_size, 
        K=args.KShot, 
        N=args.NWay, 
        Q=args.KQuery, 
        metric="supcon")
    
    trainer.test(best_model, dl_test)