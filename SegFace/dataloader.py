import os
import numpy as np
import math
import cv2
import torch
import torchvision
import random
from PIL import Image
import queue as Queue
import threading
from torch.utils.data import DataLoader, Sampler, ConcatDataset
from datasets import lapa, celebamask_hq, helen
from utils.utils import get_dist_info, worker_init_fn, DistributedSampler, get_config
from functools import partial
import os
import argparse
from dotenv import load_dotenv
load_dotenv()


def get_loader(args, dataset, split, train_bs, val_bs, test_bs, local_rank, seed, num_workers):
    rank, world_size = get_dist_info()
    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)   
    if dataset  == "lapa":
        if split == "train":
            trainset = lapa.LaPa(os.path.join(os.getenv('DATA_PATH'), "LaPa"), 'train', resolution=args.input_resolution)
            train_loader = DataLoader(dataset=trainset, batch_size=train_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(trainset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return train_loader, len(trainset.label_names)
        if split == "val":
            valset = lapa.LaPa(os.path.join(os.getenv('DATA_PATH'), "LaPa"), 'val', resolution=args.input_resolution)
            val_loader = DataLoader(dataset=valset, batch_size=val_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(valset, num_replicas=world_size, rank=local_rank, shuffle=False, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return [val_loader], ["LaPa"]
        if split == "test":
            testset = lapa.LaPa(os.path.join(os.getenv('DATA_PATH'), "LaPa"), 'test', resolution=args.input_resolution)
            test_loader = DataLoader(dataset=testset, batch_size=test_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(testset, num_replicas=world_size, rank=local_rank, shuffle=False, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return [test_loader], ["LaPa"]
    if dataset  == "celebamask_hq":
        if split == "train":
            trainset = celebamask_hq.CelebAMaskHQ(os.path.join(os.getenv('DATA_PATH'), "CelebAMask-HQ"), 'train', resolution=args.input_resolution)
            train_loader = DataLoader(dataset=trainset, batch_size=train_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(trainset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return train_loader, len(trainset.label_names)
        if split == "val":
            valset = celebamask_hq.CelebAMaskHQ(os.path.join(os.getenv('DATA_PATH'), "CelebAMask-HQ"), 'val', resolution=args.input_resolution)
            val_loader = DataLoader(dataset=valset, batch_size=val_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(valset, num_replicas=world_size, rank=local_rank, shuffle=False, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return [val_loader], ["CelebAMaskHQ"]
        if split == "test":
            testset = celebamask_hq.CelebAMaskHQ(os.path.join(os.getenv('DATA_PATH'), "CelebAMask-HQ"), 'test', resolution=args.input_resolution)
            test_loader = DataLoader(dataset=testset, batch_size=test_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(testset, num_replicas=world_size, rank=local_rank, shuffle=False, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return [test_loader], ["CelebAMaskHQ"]
    if dataset  == "helen":
        if split == "train":
            trainset = helen.HELEN(os.path.join(os.getenv('DATA_PATH'), "helen"), 'train', resolution=args.input_resolution)
            train_loader = DataLoader(dataset=trainset, batch_size=train_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(trainset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return train_loader, len(trainset.label_names)
        if split == "val":
            valset = helen.HELEN(os.path.join(os.getenv('DATA_PATH'), "helen"), 'test', resolution=args.input_resolution)
            val_loader = DataLoader(dataset=valset, batch_size=val_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(valset, num_replicas=world_size, rank=local_rank, shuffle=False, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return [val_loader], ["HELEN"]
        if split == "test":
            testset = helen.HELEN(os.path.join(os.getenv('DATA_PATH'), "helen"), 'test', resolution=args.input_resolution)
            test_loader = DataLoader(dataset=testset, batch_size=test_bs, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(testset, num_replicas=world_size, rank=local_rank, shuffle=False, seed=seed), pin_memory=True, drop_last=True, worker_init_fn=init_fn)
            return [test_loader], ["HELEN"]


if __name__ == "__main__":
    dataset = "helen"
    split = "val"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_resolution", type=int, default=512)
    args = parser.parse_args()  
    val_loaders, names = get_loader(args, dataset, split, train_bs=64, val_bs=2, test_bs=2, local_rank=0, seed=32, num_workers=4)
    loader = val_loaders[0]
    
    # Check Dataloader #
    print("Checking DataLoader...")
    for i, output in enumerate(loader):
        face = output['image'][0].permute(1, 2, 0).detach().cpu().numpy()
        face = (face * 255).astype(np.uint8)
        cv2.imwrite(f"face_{i}.png", face[:, :, ::-1])
        
        print(f"Batch {i+1}:")
        print(f"Dataset: {output['dataset']}")
        print(f"Image shape: {output['image'].shape}")
        for k in output["label"].keys():
            print(f"Label {k} shape: {output['label'][k].shape}")
        
        if i >= 2:  # Limit to first 3 batches
            break

