import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from learn2learn.algorithms import MAML
from learn2learn.vision.models import CNN4
from tqdm import tqdm
import datetime

from utils import *
from dataset_loaders import *
from encoders import *
from partition_generators import generate_unsupervised_partitions
from task_generator import TaskGenerator
from baselines.pretraining_baseline import contrastive_pretrain, test_pretrain
from baselines.metagmvae_baseline import metagmvae_train, metagmvae_test
from analyze_results.compute_dci import compute_DCI
from analyze_results.compute_partition_overlap import compute_partition_overlap


def fast_adapt(batch, inner_learner, loss_fn, num_adaptation_steps, args):
    K, K_te = args.KShot, args.KQuery
    train_data, train_labels, _, test_data, test_labels, _ = batch
    assert train_data.size(0) == K * args.NWay, f"{train_data.size(0)} VS {K * args.NWay}"
    assert test_data.size(0) == K_te * args.NWay, f"{test_data.size(0)} VS {K_te * args.NWay}"
    assert torch.numel(torch.unique(train_labels)) == torch.numel(torch.unique(test_labels)) == args.NWay
    train_data, train_labels, test_data, test_labels =  \
        train_data.to(DEVICE), train_labels.to(DEVICE), test_data.to(DEVICE), test_labels.to(DEVICE)

    # inner training (no early stopping)
    for step in range(num_adaptation_steps):
        train_loss = loss_fn(inner_learner(train_data), train_labels)
        inner_learner.adapt(train_loss)
    
    # inner testing
    test_preds = inner_learner(test_data)
    test_loss, test_accur = loss_fn(test_preds, test_labels), \
                                accuracy_fn(test_preds, test_labels)
    return test_loss, test_accur

def train(meta_model, task_generator, optimizer, loss_fn, descriptor, args):
    tb_writer = SummaryWriter(log_dir=os.path.join(LEARNCURVEDIR, descriptor))

    for i in tqdm(range(METATRAIN_OUTER_EPISODES), desc='Training Epochs'):
        optimizer.zero_grad()
        meta_train_loss, meta_train_accur = 0.0, 0.0
        for _ in range(NUM_TASKS_METATRAIN):
            # start a inner-loop learner from the current initialization parameters
            inner_learner = meta_model.clone()
            task_batch = task_generator.sample_task("meta_train", args)
            
            # inner training
            inner_test_loss, inner_test_accur = fast_adapt(task_batch,
                                                             inner_learner,
                                                             loss_fn,
                                                             METATRAIN_INNER_UPDATES,
                                                             args)
            
            # meta training update step
            inner_test_loss.backward()
            meta_train_loss += inner_test_loss.item()
            meta_train_accur += inner_test_accur.item()
        meta_train_loss /= NUM_TASKS_METATRAIN
        meta_train_accur /= NUM_TASKS_METATRAIN

        # update the meta-parameters (the initialization parameters for MAML)
        for p in meta_model.parameters():
            p.grad.data.mul_(1.0/NUM_TASKS_METATRAIN)
        optimizer.step()

        # log meta-training metrics
        tb_writer.add_scalar("Loss/meta_train", meta_train_loss, i)
        tb_writer.add_scalar("Accuracy/meta_train", meta_train_accur, i)

        # meta validation
        if (i+1) % METAVALID_OUTER_INTERVAL == 0:
            meta_valid_loss, meta_valid_accur = 0.0, 0.0
            for _ in range(NUM_TASKS_METAVALID):
                inner_learner = meta_model.clone()
                task_batch = task_generator.sample_task('meta_valid', args)
                inner_test_loss, inner_test_accur = fast_adapt(task_batch, 
                                                                inner_learner,
                                                                loss_fn,
                                                                METAVALID_INNER_UPDATES,
                                                                args)
                meta_valid_loss += inner_test_loss.item()
                meta_valid_accur += inner_test_accur.item()
            meta_valid_loss /= NUM_TASKS_METAVALID
            meta_valid_accur /= NUM_TASKS_METAVALID
        
            # log meta-validation metrics
            tb_writer.add_scalar("Loss/meta_valid", meta_valid_loss, i)
            tb_writer.add_scalar("Accuracy/meta_valid", meta_valid_accur, i)
    
    # within the call of close(), flush() should also be called upon the tensorboard writer
    tb_writer.close()
    print(f"[{descriptor}] Training function completed!")
    
    return meta_model 

def test(meta_model, task_generator, loss_fn, descriptor, args):
    meta_test_losses, meta_test_accurs = [], []
    for _ in tqdm(range(NUM_TASKS_METATEST), desc='Testing tasks'):
        inner_learner = meta_model.clone()
        task_batch = task_generator.sample_task("meta_test", args)
        inner_test_loss, inner_test_accur = fast_adapt(task_batch,
                                                       inner_learner,
                                                       loss_fn,
                                                       METATEST_INNER_UPDATES,
                                                       args)
        meta_test_losses.append(inner_test_loss.item())
        meta_test_accurs.append(inner_test_accur.item())
    
    return meta_test_accurs


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    fix_seed(args.seed)

    # Load train/validation/test datasets
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
    
    encoder = get_encoder(args, DEVICE)        
    descriptor = get_descriptor(encoder, args)

    print(f"<<<<<<<<<<Running {descriptor} on {args.dsName}...>>>>>>>>>")
    # Supervised meta-learning
    if args.encoder in ['sup', 'scratch']:
        meta_train_partitions = meta_train_partitions_supervised
    elif args.encoder == 'supall':
        meta_train_partitions = meta_train_partitions_supervised_all
    elif args.encoder == 'supora':
        meta_train_partitions = meta_train_partitions_supervised_oracle
    elif args.encoder in ['simclrpretrain', 'metagmvae']:
        # not using metatraining
        # don't let it go through unsupervised partitions generation, as it takes quite a bit of time
        meta_train_partitions = meta_train_partitions_supervised
    # Unsupervised meta-learning
    else: 
        meta_train_partitions_unsupervised = generate_unsupervised_partitions(
                                                meta_train_set, 
                                                encoder,
                                                descriptor,
                                                args)   
        meta_train_partitions = meta_train_partitions_unsupervised  

    assert meta_train_partitions

    task_generator = TaskGenerator(meta_train_set, 
                                    meta_valid_set, 
                                    meta_test_set,
                                    meta_train_partitions,
                                    meta_valid_partitions,
                                    meta_test_partitions,
                                    args)
    
    if args.visualizeTasks:
        assert args.encoder not in ["simclrpretrain", "metagmvae"]
        visualize_constructed_tasks(task_generator, descriptor, args, n_imgs=50)
        exit(0)
    elif args.computePartitionOverlap:
        compute_partition_overlap(meta_train_partitions, descriptor)
        exit(0)
    else:
        pass

    base_model = CNN4(output_size=args.NWay,
                      hidden_size=32,
                      layers=4).to(DEVICE)
    meta_model = MAML(model=base_model, lr=METATRAIN_INNER_LR, first_order=False)
    opt = optim.Adam(meta_model.parameters(), METATRAIN_OUTER_LR)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    if args.encoder != "scratch":
        model_path = os.path.join(MODELDIR, f"{descriptor}.ckpt")
        try:
            if args.encoder in ["simclrpretrain", "metagmvae"]:
                encoder.load_state_dict(torch.load(model_path))
            else:
                meta_model.load_state_dict(torch.load(model_path))
            print(f"[{descriptor}]: Loaded model from {model_path}!")
        except FileNotFoundError:
            print(f"[{descriptor}]: No model at {model_path}. Training from scratch...")
            if args.encoder == "simclrpretrain":
                opt = optim.Adam(encoder.parameters(), lr=PRETRAIN_LR)
                encoder = contrastive_pretrain(encoder, opt, meta_train_set, meta_valid_set, descriptor, args)
                torch.save(encoder.state_dict(), model_path)
            elif args.encoder == "metagmvae":
                opt = optim.Adam(encoder.parameters(), lr=GMVAE_METATRAIN_LR)
                encoder = metagmvae_train(encoder, opt, meta_train_set, meta_valid_set, descriptor, args)
                torch.save(encoder.state_dict(), model_path)
            else:
                meta_model = train(meta_model, 
                                    task_generator, 
                                    opt, 
                                    loss_fn, 
                                    descriptor, 
                                    args)
                torch.save(meta_model.state_dict(), model_path)
            print(f"Model saved at {model_path}!")

    if args.encoder == "simclrpretrain":
        meta_test_accurs = test_pretrain(encoder, task_generator, descriptor, args)
    elif args.encoder == "metagmvae":
        meta_test_accurs = metagmvae_test(encoder, task_generator, loss_fn, descriptor, args)
    else:
        meta_test_accurs = test(meta_model, task_generator, loss_fn, descriptor, args)

    with open("res.txt", "a") as f:
        f.write(str(datetime.datetime.now())+f' under seed {args.seed}'+'\n')
        f.write(f"[{descriptor} on {args.dsName} {args.NWay}-way {args.KShot}-shot meteTrain {args.KShot}-shot metaTest]: \n" + \
                f"Mean meta test accuracy: {np.mean(meta_test_accurs)*100:.2f}%\n")
    print(f"[{descriptor} on {args.dsName}] testing completed!")
    
    print("<<<<<<<<<<<<<<<Main script finished successfully!>>>>>>>>>>>>")
