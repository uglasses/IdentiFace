import os
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import torchvision
import sys
import logging
import warnings
import time
import tempfile
import config
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from torch.nn.modules.loss import CrossEntropyLoss
from torch import distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import init_logging, TimeLogging, get_config, setup_seed
from dataloader import get_loader
from inference import validation
from network import get_model
from loss import DiceLoss


rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
distributed.init_process_group("nccl")


def train(args):
    #Config & Seed Setup
    setup_seed(seed=args.seed, cuda_deterministic=False)
    #Distributed Setup
    torch.cuda.set_device(local_rank)

    #Logging Setup
    os.makedirs(os.path.join(args.ckpt_path, args.expt_name), exist_ok=True)
    if rank == 0: 
        log_root = init_logging(rank, os.path.join(args.ckpt_path, args.expt_name))
        log_root.info("---"*15)
        for arg, value in vars(args).items():
            log_root.info(f"{arg}: {value}")
        log_root.info("---"*15)

    #Dataloader
    train_dataloader, num_classes = get_loader(args, args.dataset, "train", args.train_bs, args.val_bs, args.test_bs, local_rank, args.seed, args.num_workers)
    val_dataloaders, val_names = get_loader(args, args.dataset, "val", args.train_bs, args.val_bs, args.test_bs, local_rank, args.seed, args.num_workers)
    if rank==0: log_root.info("Data Loaded")

    #Model Backbone
    model = get_model(args.backbone, args.input_resolution, args.model).cuda()
    model = DDP(module=model, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=True) 
    if rank==0: log_root.info("Model Loaded")

    #Optimizers, Scheduler & Loss Functions
    optimizer_dict = [
        {"params": model.parameters()},
    ]
    optimizer = AdamW(optimizer_dict, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=args.lr_schedule_gamma)

    # Loss Functions
    ## Segmentation Loss
    seg_ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)    

    #If resume
    start_epoch = 0
    num_iter = 0 # Max value - (len(train_dataloader)/args.train_bs * args.epochs)
    if args.resume:
        dict_checkpoint = torch.load(os.path.join(args.ckpt_path, args.expt_name,f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        num_iter = dict_checkpoint["iteration"]
        model.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        optimizer.load_state_dict(dict_checkpoint["optimizer"])
        scheduler.load_state_dict(dict_checkpoint["scheduler"])
        del dict_checkpoint

    #Logging
    if rank == 0: time_log = TimeLogging(total_iters=int(args.epochs*len(train_dataloader)), start_iter=num_iter)

    #Train Function
    start_time = time.time()
    for num_epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        train_dataloader.sampler.set_epoch(num_epoch)
        for _, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss_batch = torch.tensor(0, dtype=torch.float).cuda()
            #Feature extraction from Backbone
            images, labels, datasets = batch["image"], batch["label"], batch["dataset"] #(BS, num_classes, H, W), (BS, H, W)
            images = images.cuda()
            for k in labels.keys():
                labels[k] = labels[k].cuda()
            datasets = datasets.cuda()  
            seg_output = model(images, labels, datasets)

            loss_seg_ce = seg_ce_loss(seg_output, labels["segmentation"].to(dtype=torch.long).cuda())
            loss_dice = dice_loss(seg_output, labels["segmentation"],  softmax=True)
            loss_batch = 0.5 * loss_seg_ce + 0.5 * loss_dice

            del seg_output
            #Loss
            loss_batch.backward()
            optimizer.step()
            train_loss += loss_batch.item()

            num_iter += 1
            if num_iter % args.log_interval == 0 and rank == 0:
                log_root.info('iteration %d, epoch %d : lr %f, loss: %f, eta: %f hours' % (num_iter, num_epoch, scheduler.get_last_lr()[-1], loss_batch.item(), time_log.estimate(num_iter)))
        scheduler.step()
        train_loss /= num_iter
        if rank == 0: log_root.info(f"Epoch ({num_epoch}/{args.epochs}) | Train Epoch Loss: {train_loss}")
    
        if num_epoch % args.save_interval == 0:
            if args.save_all_states:
                checkpoint = {
                    "epoch": num_epoch,
                    "iteration": num_iter,
                    "state_dict_backbone": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(args.ckpt_path, args.expt_name,f"checkpoint_gpu_{rank}.pt"))
            
            if rank == 0:
                checkpoint = {
                    "epoch": num_epoch,
                    "iteration": num_iter,
                    "state_dict_backbone": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(args.ckpt_path, args.expt_name,f"model_{num_epoch}.pt"))

        if num_epoch % args.val_interval == 0:
            if rank == 0: 
                validation(args, model, val_dataloaders, val_names, log_root)

    torch.distributed.barrier()
    destroy_process_group()                                
    return 


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = config.get_args()  
    train(args)
    